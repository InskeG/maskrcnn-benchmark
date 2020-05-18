# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import itertools

import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
from collections import defaultdict

class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.

    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches whose
            size is less than ``batch_size``

    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven

        # for panorams dataset
        self.group_ids = torch.arange(590704)

        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)

        order[sampled_ids] = torch.arange(len(sampled_ids))

        # get a mask with the elements that were sampled
        mask = order >= 0

        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the
        # sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # permute each cluster so that they follow the order from
        # the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)



# class GroupedBatchSampler(BatchSampler):
#     """
#     Wraps another sampler to yield a mini-batch of indices.
#     It enforces that elements from the same group should appear in groups of batch_size.
#     It also tries to provide mini-batches which follows an ordering which is
#     as close as possible to the ordering from the original sampler.

#     Arguments:
#         sampler (Sampler): Base sampler.
#         batch_size (int): Size of mini-batch.
#         drop_uneven (bool): If ``True``, the sampler will drop the batches whose
#             size is less than ``batch_size``

#     """

#     def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
#         if not isinstance(sampler, Sampler):
#             raise ValueError(
#                 "sampler should be an instance of "
#                 "torch.utils.data.Sampler, but got sampler={}".format(sampler)
#             )
#         self.sampler = sampler
#         self.group_ids = torch.as_tensor(group_ids)
#         assert self.group_ids.dim() == 1
#         self.batch_size = batch_size
#         self.drop_uneven = drop_uneven

#         # for panorams dataset
#         self.group_ids = torch.arange(590704)

#         self.groups = torch.unique(self.group_ids).sort(0)[0]

#         self._can_reuse_batches = False

#     def _prepare_batches(self):
#         dataset_size = len(self.group_ids)
#         # get the sampled indices from the sampler
#         sampled_ids = torch.as_tensor(list(self.sampler))
#         # potentially not all elements of the dataset were sampled
#         # by the sampler (e.g., DistributedSampler).
#         # construct a tensor which contains -1 if the element was
#         # not sampled, and a non-negative number indicating the
#         # order where the element was sampled.
#         # for example. if sampled_ids = [3, 1] and dataset_size = 5,
#         # the order is [-1, 1, -1, 0, -1]
#         #order = torch.full((dataset_size,), -1, dtype=torch.int64)

#         #order[sampled_ids] = torch.arange(len(sampled_ids))

#         # get a mask with the elements that were sampled
#         #mask = order >= 0

#         # find the elements that belong to each individual cluster
#         clusters = [(self.group_ids == i) for i in self.groups]
#         # get relative order of the elements inside each cluster
#         # that follows the order from the sampler
#         relative_order = [order[cluster] for cluster in clusters]
#         # with the relative order, find the absolute order in the
#         # sampled space
#         permutation_ids = [s[s.sort()[1]] for s in relative_order]
#         # permute each cluster so that they follow the order from
#         # the sampler
#         permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

#         # splits each cluster in batch_size, and merge as a list of tensors
#         splits = [c.split(self.batch_size) for c in permuted_clusters]
#         merged = tuple(itertools.chain.from_iterable(splits))

#         # now each batch internally has the right order, but
#         # they are grouped by clusters. Find the permutation between
#         # different batches that brings them as close as possible to
#         # the order that we have in the sampler. For that, we will consider the
#         # ordering as coming from the first element of each batch, and sort
#         # correspondingly
#         first_element_of_batch = [t[0].item() for t in merged]
#         # get and inverse mapping from sampled indices and the position where
#         # they occur (as returned by the sampler)
#         inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
#         # from the first element in each batch, get a relative ordering
#         first_index_of_batch = torch.as_tensor(
#             [inv_sampled_ids_map[s] for s in first_element_of_batch]
#         )

#         # permute the batches so that they approximately follow the order
#         # from the sampler
#         permutation_order = first_index_of_batch.sort(0)[1].tolist()
#         # finally, permute the batches
#         batches = [merged[i].tolist() for i in permutation_order]

#         if self.drop_uneven:
#             kept = []
#             for batch in batches:
#                 if len(batch) == self.batch_size:
#                     kept.append(batch)
#             batches = kept
#         return batches

#     def __iter__(self):
#         if self._can_reuse_batches:
#             batches = self._batches
#             self._can_reuse_batches = False
#         else:
#             batches = self._prepare_batches()
#         self._batches = batches
#         return iter(batches)

#     def __len__(self):
#         if not hasattr(self, "_batches"):
#             self._batches = self._prepare_batches()
#             self._can_reuse_batches = True
#         return len(self._batches)




class GroupedBatchSamplerPanorAMS(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size