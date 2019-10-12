"""!
@brief Get some random sampling for the position of two sources


@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import numpy as np
from scipy.spatial import distance as dst
from pprint import pprint


class RandomCirclePositioner(object):
    """
    ! Returns n_source_pairs positions based on a circle with
    specified radius Cartessian and Polar coordinates like follows:

    For each pair of the list we get a dictionary of:
    {
        'thetas': angles in rads [<(+x, s1), <(+x, s2)] in list,
        'd_theta': < (+x, s2) - < (+x, s1),
        'xy_positons': [(x_1, y_1), (x_2, y_2)], Cartessian
        'distances': [[||si-mj||]] all precomputed distances
        'taus': time delays in sample format
        'amplitudes': a1 and a2 for: m2(t) = a1*s1(t+d1) + a2*s2(t+d2)
    }
    (theta_of_source_1, theta_of_source_1)



                        s2     OOO ooo
                                       OOo    (x1, y1)
               oOO
            oOO                               s1
          oOO
        oOO                                       OOo
       oOO                                         OOo
      oOO                                           OOo
     oOO                                             OOo
     oOO                                             OOo
     oOO          m1 <-- mic_distance --> m2     ================>>+x
     oOO                                             OOo
     oOO                                             OOo
      oOO                                           OOo
       oOO                                         OOo
        oOO                                       OOo
          oOO                                   OOo
            oO                                OOo
               oOO                         OOo
                   oOO                 OOo
                       ooo OOO OOO ooo
    """

    def __init__(self,
                 min_angle=np.pi/18.,
                 angle_sup=np.pi - np.pi/18.,
                 angle_inf=np.pi/18.,
                 radius=10.71,
                 mic_distance_percentage=0.002,
                 sound_speed=343,
                 fs=16000):
        """
        :param min_angle: minimum angle in rads for the 2 sources
        :param angle_sup or inf: the maximum and minimum values
        available for an angle that a source would lie on
        :param radius: Radius of the circle in **meters**
        :param mic_distance_percentage: Percentage of the radius
        corresponding to the distance between the two microphones
        :param sound_speed: Default 343 m/s in 20oC room temperature
        :param fs: sampling ratio in Hz
        """

        self.min_angle = min_angle
        self.angle_sup = angle_sup
        self.angle_inf = angle_inf
        self.radius = radius
        self.mic_distance = self.radius * mic_distance_percentage
        # for 16000 hz in order to get maximum +- 1 sample delays we
        # have to sustain a distance of maximum: 2.142 cm
        # between the mics
        self.m1 = (-self.mic_distance / 2, 0.)
        self.m2 = (self.mic_distance / 2, 0.)
        self.sound_speed = sound_speed
        self.fs = fs

    @staticmethod
    def get_cartessian_position(radius,
                                angle):
        return radius * np.cos(angle), radius * np.sin(angle)

    def get_amplifier_values_for_sources(self,
                                         n_sources):
        """
        :return: A dictionary of all the amplitudes in order to infer
        the final mixture depending on the weighted summation of the
        source-signals
        """
        alphas = np.random.uniform(low=0.2,
                                   high=1.0,
                                   size=n_sources)
        total_amplitude = sum(alphas)

        return dict([("a"+str(i+1), a/total_amplitude)
                     for (i, a) in enumerate(alphas)])

    def get_time_delays_for_sources(self,
                                    distances,
                                    n_sources):
        # delays are always computed using the m1 microphone as
        # reference and comparing to the time delay from m2

        taus_list = []
        for i in np.arange(n_sources):
            source = "s"+str(i+1)
            taus_list.append(distances[source+"m1"]
                             - distances[source+"m2"])

        return [(1. * self.fs * tau) / self.sound_speed
                for tau in taus_list]

    def compute_distances_for_sources_and_mics(self,
                                               source_points):
        """! si \in source_points must be in format (xi, yi)
        \:return a dictionary of all given points"""
        points = {"m1": self.m1, "m2": self.m2}
        points.update(dict([("s"+str(i+1), xy)
                            for (i, xy) in enumerate(source_points)]))
        distances = {}

        for point_1, xy1 in points.items():
            for point_2, xy2 in points.items():
                distances[point_1+point_2] = dst.euclidean(xy1, xy2)

        return distances

    def get_angles(self, n_source_pairs):
        while True:
            thetas = np.random.uniform(low=self.angle_inf,
                                       high=self.angle_sup,
                                       size=n_source_pairs)
            thetas = sorted(thetas)
            d_thetas = [th2 - th1 for (th1, th2) in
                        zip(thetas[:-1], thetas[1:])]

            min_angle_enforced = np.where(np.abs(d_thetas) <
                                    self.min_angle)[0].shape[0] == 0

            if min_angle_enforced:
                break

        return thetas, d_thetas

    def get_sources_locations(self,
                              n_source_pairs):
        """!
        Generate the positions, angles and distances for
        n_source_pairs of the same mixture corersponding to 2 mics"""
        thetas, d_thetas = self.get_angles(n_source_pairs)
        xys = []
        for angle in thetas:
            xys.append(self.get_cartessian_position(self.radius, angle))

        distances = self.compute_distances_for_sources_and_mics(xys)

        taus = self.get_time_delays_for_sources(distances,
                                                n_source_pairs)

        mix_amplitudes = self.get_amplifier_values_for_sources(
                              n_source_pairs)

        sources_locations = {'thetas': np.asarray(thetas),
                             'd_thetas': np.asarray(d_thetas),
                             'xy_positons': np.asarray(xys),
                             'distances': distances,
                             'taus': np.asarray(taus),
                             'amplitudes': np.asarray(list(
                                           mix_amplitudes.values()))}

        return sources_locations


def example_of_usage():
    """
    :return:
    {'amplitudes': array([0.28292362, 0.08583346, 0.63124292]),
     'd_thetas': array([1.37373734, 1.76785531]),
     'distances': {'m1m1': 0.0,
                   'm1m2': 0.03,
                   'm1s1': 3.015, ...
                   's3s3': 0.0},
     'taus': array([ 1, -1,  0]),
     'thetas': array([0.        , 1.37373734, 3.14159265]),
     'xy_positons': array([[ 3.00000000e+00,  0.00000000e+00],
           [ 5.87358252e-01,  2.94193988e+00],
           [-3.00000000e+00,  3.67394040e-16]])}
    """
    random_positioner = RandomCirclePositioner()
    positions_info = random_positioner.get_sources_locations(5)
    pprint(positions_info)


if __name__ == "__main__":
    example_of_usage()
