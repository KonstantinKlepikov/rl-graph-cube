import torch


class BeamSearch:
    """Beam search

    TODO: docstrings

    """

    def __init__(
        self,
        state_start: list[int] | torch.Tensor,
        generators: list[int] | torch.Tensor,
        state_dest: torch.Tensor | None = None,
        beam_width: int = 2,
        n_steps_limit: int = 100,
        hasher: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> None:
        self._device = self._set_devise()
        self._generators, self._state_size = self._get_generators_and_size(generators)
        self._dtype = self._get_dtype()
        self._state_dest = self._get_state_dest(state_dest)
        self._state_start = self._get_state_start(state_start)
        self._hasher = self._get_hasher(hasher)
        self.n_steps_limit = n_steps_limit
        self.verbose = verbose
        self.beam_width = beam_width
        self.is_found: bool = False
        self.step: int | None = None
        self._log: list[str] = []

    def _set_devise(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _get_generators_and_size(
        self,
        generators: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(generators, list):
            lgen = generators
        elif isinstance(generators, torch.Tensor):
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
        return torch.gather(
            states.unsqueeze(1).expand(states.size(0), moves.shape[0], states.size(1)),
            2,
            moves.unsqueeze(0).expand(states.size(0), moves.shape[0], states.size(1)),
        ).flatten(end_dim=1)

    def _get_unique_states(
        self,
        states: torch.Tensor,
        vec_hasher: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return matrix with unique rows for input matrix "states"
        duplicate rows are dropped.

        NOTE: output order of rows is different from the original.
        """
        # Hash
        hashed = torch.sum(states * vec_hasher.to(self._device), dim=1)

        # Sort
        hashed_sorted, idx = torch.sort(hashed)

        # Unique elements
        mask = torch.concat(
            (
                torch.tensor([True], device=self._device),
                hashed_sorted[1:] - hashed_sorted[:-1] > 0,
            )
        )
        return states[idx][mask]

    def _init_array_of_state(self) -> torch.Tensor:
        return (
            self._state_start.view(-1, self._state_size)
            .clone()
            .to(self._dtype)
            .to(self._device)
        )

    def _hamming_distance(self, aofs_new: torch.Tensor) -> torch.Tensor:
        return torch.sum((aofs_new == self._state_dest[0, :]), dim=1)

    def _search(self, estimator) -> None:

        aofs = self._init_array_of_state()

        for i_step in range(1, self.n_steps_limit + 1):

            # Apply generator to all current states
            aofs_new = self._get_neighbors(aofs, self._generators.to(torch.int64))

            # Take only unique states
            aofs_new = self._get_unique_states(aofs_new, self._hasher)

            # Check destination state found
            vec_tmp = torch.all(aofs_new == self._state_dest, axis=1)
            self.is_found = bool(torch.any(vec_tmp).item())
            if self.is_found:
                self.step = i_step
                if self.verbose:
                    print(
                        f'Found destination state. Step: {i_step} Ways:'
                        f' {(vec_tmp).sum()}.'
                    )
                break

            # Estimate distance of new states to the destination state
            if aofs_new.shape[0] > self.beam_width:

                estimation = estimator(aofs_new)

                # Take only "beam_width" of the best states
                idx = torch.argsort(estimation)[: self.beam_width]
                aofs = aofs_new[idx, :]

            else:

                aofs = aofs_new

            if self.verbose:
                print(f'Step: {i_step}. Shape: {aofs_new.shape}.')

        if self.verbose:
            self.print_result()

    def dummy_search(self) -> None:
        """Search with Hamming distance"""
        self._search(self._hamming_distance)

    def print_result(self) -> None:
        print(f'Search finished. Beam_width: {self.beam_width}')
        if self.is_found:
            print(f'{self.step} steps to destination state. Path found.')
        else:
            print('Path not found.')
        print('-' * 5)


if __name__ == '__main__':

    bs = BeamSearch(
        state_start=[1, 0],
        generators=[[1, 0]],
    )
    bs.dummy_search()
    bs.print_result()

    bs = BeamSearch(
        state_start=[2, 1, 0],
        generators=[[1, 0, 2], [0, 2, 1]],
        verbose=True,
    )
    bs.dummy_search()
