# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from .track import Track
from scipy.optimize import linear_sum_assignment

INFTY_COST = 1e+5

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        # 1. For matching results
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        # 2. Call mark_missed for unmatched tracker
        # track mismatch, delete if pending, delete if update time is too long
        # max age is a lifetime, default 30 frames
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        # The main function is to match, find the matched and unmatched parts
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # Function: used to calculate the distance between track and detection, cost function
            # Need to be used before KM algorithm
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # Calculate the cost matrix cosine distance through the nearest neighbor
            cost_matrix = self.metric.distance(features, targets)
            # Calculate Mahalanobis distance and get a net state matrix
            cost_matrix = gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            self.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            self.min_cost_matching(
                self.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_name = detection.get_class()
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, class_name, class_confidence=detection.confidence))
        self._next_id += 1

    def min_cost_matching(self,
            distance_metric, max_distance, tracks, detections, track_indices=None,
            detection_indices=None):
        """Solve linear assignment problem.

        Parameters
        ----------
        distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
            The distance metric is given a list of tracks and detections as well as
            a list of N track indices and M detection indices. The metric should
            return the NxM dimensional cost matrix, where element (i, j) is the
            association cost between the i-th track in the given track indices and
            the j-th detection in the given detection_indices.
        max_distance : float
            Gating threshold. Associations with cost larger than this value are
            disregarded.
        tracks : List[track.Track]
            A list of predicted tracks at the current time step.
        detections : List[detection.Detection]
            A list of detections at the current time step.
        track_indices : List[int]
            List of track indices that maps rows in `cost_matrix` to tracks in
            `tracks` (see description above).
        detection_indices : List[int]
            List of detection indices that maps columns in `cost_matrix` to
            detections in `detections` (see description above).

        Returns
        -------
        (List[(int, int)], List[int], List[int])
            Returns a tuple with the following three entries:
            * A list of matched track and detection indices.
            * A list of unmatched track indices.
            * A list of unmatched detection indices.

        """
        if track_indices is None:
            track_indices = np.arange(len(tracks))
        if detection_indices is None:
            detection_indices = np.arange(len(detections))

        if len(detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, detection_indices  # Nothing to match.

        cost_matrix = distance_metric(
            tracks, detections, track_indices, detection_indices)
        cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
        indices = linear_sum_assignment(cost_matrix)
        indices = np.asarray(indices)
        indices = np.transpose(indices)
        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection_idx in enumerate(detection_indices):
            if col not in indices[:, 1]:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(track_indices):
            if row not in indices[:, 0]:
                unmatched_tracks.append(track_idx)
        for row, col in indices:
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            if cost_matrix[row, col] > max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        return matches, unmatched_tracks, unmatched_detections

    def matching_cascade(self,
            distance_metric, max_distance, cascade_depth, tracks, detections,
            track_indices=None, detection_indices=None):
        """Run matching cascade.

        Parameters
        ----------
        distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
            The distance metric is given a list of tracks and detections as well as
            a list of N track indices and M detection indices. The metric should
            return the NxM dimensional cost matrix, where element (i, j) is the
            association cost between the i-th track in the given track indices and
            the j-th detection in the given detection indices.
        max_distance : float
            Gating threshold. Associations with cost larger than this value are
            disregarded.
        cascade_depth: int
            The cascade depth, should be se to the maximum track age.
        tracks : List[track.Track]
            A list of predicted tracks at the current time step.
        detections : List[detection.Detection]
            A list of detections at the current time step.
        track_indices : Optional[List[int]]
            List of track indices that maps rows in `cost_matrix` to tracks in
            `tracks` (see description above). Defaults to all tracks.
        detection_indices : Optional[List[int]]
            List of detection indices that maps columns in `cost_matrix` to
            detections in `detections` (see description above). Defaults to all
            detections.

        Returns
        -------
        (List[(int, int)], List[int], List[int])
            Returns a tuple with the following three entries:
            * A list of matched track and detection indices.
            * A list of unmatched track indices.
            * A list of unmatched detection indices.

        """
        if track_indices is None:
            track_indices = list(range(len(tracks)))
        if detection_indices is None:
            detection_indices = list(range(len(detections)))

        unmatched_detections = detection_indices
        matches = []
        for level in range(cascade_depth):
            if len(unmatched_detections) == 0:  # No detections left
                break

            track_indices_l = [
                k for k in track_indices
                if tracks[k].time_since_update == 1 + level
            ]
            if len(track_indices_l) == 0:  # Nothing to match at this level
                continue

            matches_l, _, unmatched_detections = \
                self.min_cost_matching(
                    distance_metric, max_distance, tracks, detections,
                    track_indices_l, unmatched_detections)
            matches += matches_l
        unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
        return matches, unmatched_tracks, unmatched_detections

    def gate_cost_matrix(self,
            kf, cost_matrix, tracks, detections, track_indices, detection_indices,
            gated_cost=INFTY_COST, only_position=False):
        """Invalidate infeasible entries in cost matrix based on the state
        distributions obtained by Kalman filtering.

        Parameters
        ----------
        kf : The Kalman filter.
        cost_matrix : ndarray
            The NxM dimensional cost matrix, where N is the number of track indices
            and M is the number of detection indices, such that entry (i, j) is the
            association cost between `tracks[track_indices[i]]` and
            `detections[detection_indices[j]]`.
        tracks : List[track.Track]
            A list of predicted tracks at the current time step.
        detections : List[detection.Detection]
            A list of detections at the current time step.
        track_indices : List[int]
            List of track indices that maps rows in `cost_matrix` to tracks in
            `tracks` (see description above).
        detection_indices : List[int]
            List of detection indices that maps columns in `cost_matrix` to
            detections in `detections` (see description above).
        gated_cost : Optional[float]
            Entries in the cost matrix corresponding to infeasible associations are
            set this value. Defaults to a very large value.
        only_position : Optional[bool]
            If True, only the x, y position of the state distribution is considered
            during gating. Defaults to False.

        Returns
        -------
        ndarray
            Returns the modified cost matrix.

        """
        gating_dim = 2 if only_position else 4
        gating_threshold = kalman_filter.chi2inv95[gating_dim]
        measurements = np.asarray(
            [detections[i].to_xyah() for i in detection_indices])
        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            gating_distance = kf.gating_distance(
                track.mean, track.covariance, measurements, only_position)
            cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        return cost_matrix

    def iou(self, bbox, candidates):
        """Computer intersection over union.

        Parameters
        ----------
        bbox : ndarray
            A bounding box in format `(top left x, top left y, width, height)`.
        candidates : ndarray
            A matrix of candidate bounding boxes (one per row) in the same format
            as `bbox`.

        Returns
        -------
        ndarray
            The intersection over union in [0, 1] between the `bbox` and each
            candidate. A higher score means a larger fraction of the `bbox` is
            occluded by the candidate.

        """
        bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
        candidates_tl = candidates[:, :2]
        candidates_br = candidates[:, :2] + candidates[:, 2:]

        tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                   np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
        br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                   np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
        wh = np.maximum(0., br - tl)

        area_intersection = wh.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)
        return area_intersection / (area_bbox + area_candidates - area_intersection)

    def iou_cost(self, tracks, detections, track_indices=None,
                 detection_indices=None):
        """An intersection over union distance metric.

        Parameters
        ----------
        tracks : List[deep_sort.track.Track]
            A list of tracks.
        detections : List[deep_sort.detection.Detection]
            A list of detections.
        track_indices : Optional[List[int]]
            A list of indices to tracks that should be matched. Defaults to
            all `tracks`.
        detection_indices : Optional[List[int]]
            A list of indices to detections that should be matched. Defaults
            to all `detections`.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape
            len(track_indices), len(detection_indices) where entry (i, j) is
            `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

        """
        if track_indices is None:
            track_indices = np.arange(len(tracks))
        if detection_indices is None:
            detection_indices = np.arange(len(detections))

        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        for row, track_idx in enumerate(track_indices):
            if tracks[track_idx].time_since_update > 1:
                cost_matrix[row, :] = INFTY_COST
                continue

            bbox = tracks[track_idx].to_tlwh()
            candidates = np.asarray([detections[i].tlwh for i in detection_indices])
            cost_matrix[row, :] = 1. - self.iou(bbox, candidates)
        return cost_matrix

