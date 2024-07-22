cimport cython

from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion,
                                               TRACKPOINT,
                                               ENDPOINT,
                                               OUTSIDEIMAGE,
                                               INVALIDPOINT,)
from dipy.utils.fast_numpy cimport copy_point


cimport numpy as cnp

cdef extern from "dpy_math.h" nogil:
    int dpy_signbit(double x)
    double dpy_rint(double x)
    double fabs(double)

@cython.cdivision(True)
cdef inline double _stepsize(double point, double increment) nogil:
    """Compute the step size to the closest boundary in units of increment."""
    cdef:
        double dist
    dist = dpy_rint(point) + .5 - dpy_signbit(increment) - point
    if dist == 0:
        # Point is on an edge, return step size to next edge.  This is most
        # likely to come up if overstep is set to 0.
        return 1. / fabs(increment)
    else:
        return dist / increment

cdef void _step_to_boundary(double * point, double * direction,
                           double overstep) nogil:
    """Takes a step from point in along direction just past a voxel boundary.

    Parameters
    ----------
    direction : c-pointer to double[3]
        The direction along which the step should be taken.
    point : c-pointer to double[3]
        The tracking point which will be updated by this function.
    overstep : double
        It's often useful to have the points of a streamline lie inside of a
        voxel instead of having them lie on the boundary. For this reason,
        each step will overshoot the boundary by ``overstep * direction``.
        This should not be negative.

    """
    cdef:
        double step_sizes[3]
        double smallest_step

    for i in range(3):
        step_sizes[i] = _stepsize(point[i], direction[i])

    smallest_step = step_sizes[0]
    for i in range(1, 3):
        if step_sizes[i] < smallest_step:
            smallest_step = step_sizes[i]

    smallest_step += overstep
    for i in range(3):
        point[i] += smallest_step * direction[i]

cdef void _fixed_step(double * point, double * direction, double step_size) nogil:
    """Updates point by stepping in direction.

    Parameters
    ----------
    direction : c-pointer to double[3]
        The direction along which the step should be taken.
    point : c-pointer to double[3]
        The tracking point which will be updated by this function.
    step_size : double
        The size of step in units of direction.

    """
    for i in range(3):
        point[i] += direction[i] * step_size



cdef class DirectionGetter:

    cpdef cnp.ndarray[cnp.float_t, ndim=2] initial_direction(
            self, double[::1] point):
        pass


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef tuple generate_streamline(self,
                                    double[::1] seed,
                                    double[::1] direction,
                                    double[::1] voxel_size,
                                    double step_size,
                                    StoppingCriterion stopping_criterion,
                                    cnp.float_t[:, :] streamline,
                                    StreamlineStatus stream_status,
                                    int fixedstep
                                    ):
        cdef:
            cnp.npy_intp i
            cnp.npy_intp len_streamlines = streamline.shape[0]
            double point[3]
            double voxdir[3]
            void (*step)(double*, double*, double) nogil

        if fixedstep > 0:
            step = _fixed_step
        else:
            step = _step_to_boundary

        copy_point(&seed[0], point)
        copy_point(&seed[0], &streamline[0,0])

        stream_status = TRACKPOINT
        for i in range(1, len_streamlines):
            if i != 1 and self.get_direction_c(point, &direction[0]):  # don't get direction for first iteration
                break
            for j in range(3):
                voxdir[j] = direction[j] / voxel_size[j]
            step(point, voxdir, step_size)
            copy_point(point, &streamline[i, 0])
            stream_status = stopping_criterion.check_point_c(point)
            if stream_status == TRACKPOINT:
                continue
            elif (stream_status == ENDPOINT or
                  stream_status == INVALIDPOINT or
                  stream_status == OUTSIDEIMAGE):
                break
        else:
            # maximum length of streamline has been reached, return everything
            i = streamline.shape[0]
        return i, stream_status

    cpdef int get_direction(self,
                            double[::1] point,
                            double[::1] direction) except -1:
        return self.get_direction_c(&point[0], &direction[0])

    cdef int get_direction_c(self, double* point, double* direction):
        pass


# SB 7/22/2024
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple generate_streamline_rk4(DirectionGetter dg,
                                    double[::1] seed,
                                    double[::1] direction,
                                    double[::1] voxel_size,
                                    double step_size,
                                    StoppingCriterion stopping_criterion,
                                    cnp.float_t[:, :] streamline,
                                    StreamlineStatus stream_status,
                                    int fixedstep
                                    ):
    cdef:
        cnp.npy_intp i
        cnp.npy_intp len_streamlines = streamline.shape[0]
        double point[3]
        double midpoint[3]  # 'midpoint' for runge kutta tracking
        double k1_unit_vox[3]  # k1 in voxel coordinates, unit in world coordinates
        double k2_unit_vox[3]  # k2 in voxel coordinates, unit in world coordinates
        double k3_unit_vox[3]  # k3 in voxel coords, unit in world
        double k4_unit_vox[3]  # k4 in voxel coords, unit in world
        double k_vox_sum_dir[3]  # (1/6) * (k1 + (2 * k2) + (2 * k3) + k4)
        double world_unit_dir[3]  # normalized k_sum_dir
        double voxdir[3]  # final step vector in unit vox coords
        void (*step)(double*, double*, double) nogil

    if fixedstep > 0:
        step = _fixed_step
    else:
        step = _step_to_boundary

    copy_point(&seed[0], point)
    copy_point(&seed[0], &streamline[0,0])

    stream_status = TRACKPOINT
    for i in range(1, len_streamlines):
        if i != 1 and dg.get_direction_c(point, &direction[0]):  # don't get direction for first iteration
            break

        # RK first iteration
        for j in range(3):  # copy first direction into k1
            k1_unit_vox[j] = direction[j] / voxel_size[j]
        copy_point(point, midpoint)  # set midpoint=point for stepping
        step(midpoint, k1_unit_vox, step_size/2.0)  # half step size along k1 to first midpoint
        k1_midpoint_status = stopping_criterion.check_point_c(midpoint)
        if k1_midpoint_status != TRACKPOINT:  # if can't get direction from k1 midpoint, must stop tracking
            break
        
        # RK second iteration - direction holds k1 in world scale
        if dg.get_direction_c(midpoint, &direction[0]):  # break if cant get direction
            break
        for j in range(3):
            k2_unit_vox[j] = direction[j] / voxel_size[j]
        copy_point(point, midpoint)  # reset midpoint=point for stepping
        step(midpoint, k2_unit_vox, step_size/2.0)  # half step along k2 to second midpoint
        k2_midpoint_status = stopping_criterion.check_point_c(midpoint)
        if k2_midpoint_status != TRACKPOINT:  # k2 midpoint is out of mask, stop tracking
            break

        # RK third iteration
        if dg.get_direction_c(midpoint, &direction[0]):  # break if cant get direction
            break
        for j in range(3):
            k3_unit_vox[j] = direction[j] / voxel_size[j]
        copy_point(point, midpoint)  # reset midpoint=point
        step(midpoint, k3_unit_vox, step_size)  # full step along k3 to get point at which k4 is taken
        k3_endpoint_status = stopping_criterion.check_point_c(midpoint)
        if k3_endpoint_status != TRACKPOINT:  # if k3 endpoint is out of mask, stop tracking
            break
        
        # RK final iteration and calculation
        if dg.get_direction_c(midpoint, &direction[0]):  # if cant get direction at endpoint of k3 propagation, stop tracking
            break
        for j in range(3):  # update k4 in vox coords, and get weighted sum of k directions in voxel coords
            k4_unit_vox[j] = direction[j] / voxel_size[j]
            k_vox_sum_dir[j] = (1.0/6.0) * (k1_unit_vox[j] + (2.0 * k2_unit_vox[j]) + (2.0 * k3_unit_vox[j]) + k4_unit_vox[j])
        for j in range(3):
            world_unit_dir[j] = k_vox_sum_dir[j] * voxel_size[j]
        for j in range(3):
            world_unit_dir[j] = world_unit_dir[j] / (pow(pow(world_unit_dir[0], 2.0) + pow(world_unit_dir[1], 2.0) + pow(world_unit_dir[2], 2.0), 0.5))
        for j in range(3):
            voxdir[j] = world_unit_dir[j] / voxel_size[j]

        step(point, voxdir, step_size)
        copy_point(point, &streamline[i, 0])
        stream_status = stopping_criterion.check_point_c(point)
        if stream_status == TRACKPOINT:
            continue
        elif (stream_status == ENDPOINT or
                stream_status == INVALIDPOINT or
                stream_status == OUTSIDEIMAGE):
            break
    else:
        # maximum length of streamline has been reached, return everything
        i = streamline.shape[0]
    return i, stream_status
