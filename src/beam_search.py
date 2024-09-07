import time

import numpy as np
import torch


class BeamSearch:
    """Beam search"""

    def __init__(
        self,
        state_start: list[int] | torch.Tensor,
        generators: list[int] | torch.Tensor | np.ndarray,
        models_or_heuristics: str = 'Hamming',
        beam_width: int = 2,
        state_dest: torch.Tensor | None = None,
        n_steps_limit: int = 100,
        hasher: torch.Tensor | None = None,
        verbose: int = 0,
    ) -> None:

        if models_or_heuristics != 'Hamming':
            raise ValueError(
                f'Unsupported models_or_heauristics {models_or_heuristics}'
            )

        self._device = self._set_devise()
        print(f'Device: {self._device}\n')
        self._generators, self._state_size = self._get_generators_and_size(generators)
        self._dtype = self._get_dtype()
        self._state_dest = self._get_state_dest(state_dest)
        self._state_start = self._get_state_start(state_start)
        self._hasher = self._get_hasher(hasher)
        self.n_steps_limit = n_steps_limit
        self.verbose = verbose
        self.beam_width = beam_width

    def _set_devise(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _get_generators_and_size(
        self,
        generators: list[int] | torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        if isinstance(generators, list):
            lgen = generators
        elif isinstance(generators, torch.Tensor):
            lgen = [list(generators[i, :]) for i in range(generators.shape[0])]
        elif isinstance(generators, np.ndarray):
            lgen = [list(generators[i, :]) for i in range(generators.shape[0])]

        return torch.tensor(data=lgen, device=self._device, dtype=torch.int64), len(
            lgen[0]
        )

    def _get_dtype(self) -> torch.dtype:
        if self._state_size <= 256:
            return torch.uint8
        return torch.uint16

    def _get_state_dest(self, state_dest: torch.Tensor | None) -> torch.Tensor:
        if state_dest is not None:
            return (
                state_dest.to(self._device)
                .to(self._dtype)
                .reshape(-1, self._state_size)
            )
        return torch.arange(
            self._state_size,
            device=self._device,
            dtype=self._dtype,
        ).reshape(-1, self._state_size)

    def _get_state_start(self, state_start: list[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(state_start, torch.Tensor):
            return (
                state_start.to(self._device)
                .to(self._dtype)
                .reshape(-1, self._state_size)
            )
        return torch.tensor(
            state_start,
            device=self._device,
            dtype=self._dtype,
        ).reshape(-1, self._state_size)

    def _get_hasher(self, hasher: torch.Tensor | None):
        dtype_for_hash = torch.int64
        if hasher is None:
            max_int = int((2**62))
            return torch.randint(
                -max_int,
                max_int,
                size=(self._state_size,),
                device=self._device,
                dtype=dtype_for_hash,
            )
        return hasher.to(self._device).to(dtype_for_hash)

    @staticmethod
    def _get_neighbors(states: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        """
        Some torch magic to calculate all new states which
        can be obtained from states by moves
        """
        return torch.gather(
            states.unsqueeze(1).expand(states.size(0), moves.shape[0], states.size(1)),
            2,
            moves.unsqueeze(0).expand(states.size(0), moves.shape[0], states.size(1)),
        )

    def _get_unique_states(
        self,
        states: torch.Tensor,
        vec_hasher: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return matrix with unique rows for input matrix "states"
        I.e. duplicate rows are dropped.
        For fast implementation: we use hashing via scalar/dot product.
        Note: output order of rows is different from the original.

        NOTE: that implementation is 30 times faster than
        torch.unique(states, dim = 0) - because we use hashes
        (see K.Khoruzhii: https://t.me/sberlogasci/10989/15920)

        NOTE: torch.unique does not support returning of indices of unique element
        so we cannot use it
        That is in contrast to numpy.unique which supports - set: return_index = True
        """
        t1 = time.time()
        # Hashing rows of states matrix:
        hashed = torch.sum(states * vec_hasher.to(self._device), dim=1)
        # It is same as matrix product torch.matmul(hash_vec , states )
        # but pay attention: such code work with GPU for integers
        # While torch.matmul - does not work for GPU for integer data types,
        # since old GPU hardware (before 2020: P100, T4) does not support integer
        # matrix multiplication
        t1 = time.time() - t1
        print(t1, 'hash')

        # Sort
        t1 = time.time()
        hashed_sorted, idx = torch.sort(hashed)
        t1 = time.time() - t1
        print(t1, 'sort')

        # Mask selects elements which are different from the consequite
        # that is unique elements (since vector is sorted on the previous step)
        t1 = time.time()
        mask = torch.concat(
            (
                torch.tensor([True], device=self._device),
                hashed_sorted[1:] - hashed_sorted[:-1] > 0,
            )
        )
        t1 = time.time() - t1
        print(t1, 'mask')
        return states[idx][mask]

    def permutations(self):  # noqa
        """"""

        # Initialize array of states
        array_of_states = (
            self._state_start.view(-1, self._state_size)
            .clone()
            .to(self._dtype)
            .to(self._device)
        )

        # Main Loop over steps
        for i_step in range(1, self.n_steps_limit + 1):

            # Apply generator to all current states
            array_of_states_new = self._get_neighbors(
                array_of_states,
                self._generators.to(torch.int64),
            ).flatten(end_dim=1)

            # Take only unique states
            # surprise: THAT IS CRITICAL for beam search performance !!!!
            # if that is not done - beam search
            # will not find the desired state - quite often
            # The reason - essentianlly beam can degrade, i.e. can be populated
            # by copy of only one state
            # It is surprising that such degradation  happens quite often even for
            # beam_width = 10_000 - but it is indeed so
            array_of_states_new = self._get_unique_states(
                array_of_states_new, self._hasher
            )

            # Check destination state found
            vec_tmp = torch.all(
                array_of_states_new == self._state_dest, axis=1
            )   # Compare state_destination and each row array_of_states
            flag_found_destination = torch.any(vec_tmp).item()   # Check for coincidence
            if flag_found_destination:
                if self.verbose >= 10:
                    print(
                        'Found destination state. ',
                        'i_step:',
                        i_step,
                        ' n_ways:',
                        (vec_tmp).sum(),
                    )
                break

            # Estimate distance of new states to the destination state
            # (or best moves probabilities for policy models)
            # If we have not so many states - we take them all - no need for ML-model
            if array_of_states_new.shape[0] > self.beam_width:
                # Hamming
                estimations_for_new_states = torch.sum(
                    (array_of_states_new == self._state_dest[0, :]), dim=1
                )

                # Take only "beam_width" of the best states
                # (i.e. most nearest to destination according to the model estimate)
                idx = torch.argsort(estimations_for_new_states)[: self.beam_width]
                array_of_states = array_of_states_new[idx, :]

            else:
                # If number of states is less than beam_width - we take them all:
                array_of_states = array_of_states_new

            if self.verbose >= 10:
                print(
                    i_step,
                    'i_step',
                    array_of_states_new.shape,
                    'array_of_states_new.shape',
                )

        dict_additional_data = {}
        if self.verbose >= 1:
            print()
            print('Search finished.', 'beam_width:', self.beam_width)
            if flag_found_destination:
                print(i_step, ' steps to destination state. Path found.')
            else:
                print('Path not found.')

        return flag_found_destination, i_step, dict_additional_data


if __name__ == '__main__':

    bs = BeamSearch(state_start=[1, 0], generators=[[1, 0]], verbose=1)
    flag_found_destination, i_step, dict_additional_data = bs.permutations()

    bs = BeamSearch(state_start=[2, 1, 0], generators=[[1, 0, 2], [0, 2, 1]], verbose=1)
    flag_found_destination, i_step, dict_additional_data = bs.permutations()
