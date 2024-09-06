import time

import numpy as np
import torch


def beam_search_permutations_torch(
    state_start: list[int],
    generators: list[int] | torch.Tensor | np.ndarray,
    models_or_heuristics='Hamming',
    beam_width=2,
    state_dest: torch.Tensor | None = None,
    n_steps_limit=100,
    vec_hasher='Auto',
    verbose=0,
):
    """
    Find path from the "state_start" to the "state_destination" via beam search.

    Main parameters:
        state_start - state to be solved, i.e.
                      from where we need to find path to the destination
        generators - generators of the group
        beam_width - beam width
        state_destination = '01234...' - destination state, typically 0,1,2,3,... -
                                         identity permutation
        models_or_heuristics - machine learning model or name for hearistical metric
        n_step_max - maximal number of steps to try
    Technical parameters:
        vec_hasher - vector used for hashing
        verbose    - contols how many text output during the exection
    """
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'{device=}')

    # Analyse input params and convert to stadard forms

    # generators_type = 'permutation' # 'matrix'

    if isinstance(generators, list):
        list_generators = generators
    elif isinstance(generators, torch.Tensor):
        list_generators = [list(generators[i, :]) for i in range(generators.shape[0])]
    elif isinstance(generators, np.ndarray):
        list_generators = [list(generators[i, :]) for i in range(generators.shape[0])]

    state_size = len(list_generators[0])
    tensor_all_generators = torch.tensor(
        data=list_generators,
        device=device,
        dtype=torch.int64,
    )

    if state_size <= 256:
        dtype = torch.uint8
    else:
        dtype = torch.uint16

    # Destination state
    if state_dest is not None:
        state_dest = state_dest.to(device).to(dtype).reshape(-1, state_size)
    else:
        state_dest = torch.arange(
            state_size,
            device=device,
            dtype=dtype,
        ).reshape(-1, state_size)
    # else:
    #     state_dest = torch.tensor(
    #         state_dest, device=device, dtype=dtype
    #     ).reshape(-1, state_size)

    # state_start
    if isinstance(state_start, torch.Tensor):
        state_start = state_start.to(device).to(dtype).reshape(-1, state_size)
    else:
        state_start = torch.tensor(state_start, device=device, dtype=dtype).reshape(
            -1, state_size
        )

    # Vec_hasher
    dtype_for_hash = torch.int64
    if vec_hasher == 'Auto':
        # Hash vector generation
        max_int = int((2**62))     # print(max_int)
        vec_hasher = torch.randint(
            -max_int, max_int, size=(state_size,), device=device, dtype=dtype_for_hash
        )   #
    elif not isinstance(vec_hasher, torch.Tensor):
        vec_hasher = torch.tensor(vec_hasher, device=device, dtype=dtype_for_hash)
    else:
        vec_hasher = vec_hasher.to(device).to(dtype_for_hash)

    ##########################################################################################
    # Initializations
    ##########################################################################################

    # Initialize array of states
    array_of_states = state_start.view(-1, state_size).clone().to(dtype).to(device)

    ##########################################################################################
    # Main Loop over steps
    ##########################################################################################
    for i_step in range(1, n_steps_limit + 1):

        # Apply generator to all current states
        array_of_states_new = get_neighbors(
            array_of_states, tensor_all_generators.to(torch.int64)
        ).flatten(end_dim=1)

        # Take only unique states
        # surprise: THAT IS CRITICAL for beam search performance !!!!
        # if that is not done - beam search  will not find the desired state - quite often
        # The reason - essentianlly beam can degrade, i.e. can be populated by copy of only one state
        # It is surprising that such degradation  happens quite often even for beam_width = 10_000 - but it is indeed so
        array_of_states_new = get_unique_states_2(array_of_states_new, vec_hasher)

        # Check destination state found
        vec_tmp = torch.all(
            array_of_states_new == state_destination, axis=1
        )   # Compare state_destination and each row array_of_states
        flag_found_destination = torch.any(vec_tmp).item()   # Check for coincidence
        if flag_found_destination:
            if verbose >= 10:
                print(
                    'Found destination state. ',
                    'i_step:',
                    i_step,
                    ' n_ways:',
                    (vec_tmp).sum(),
                )
            break

        # Estimate distance of new states to the destination state (or best moves probabilities for policy models)
        if (
            array_of_states_new.shape[0] > beam_width
        ):   # If we have not so many states - we take them all - no need for ML-model
            if models_or_heuristics == 'Hamming':
                estimations_for_new_states = torch.sum(
                    (array_of_states_new == state_destination[0, :]), dim=1
                )
            else:
                raise ValueError(
                    'Unsupported models_or_heauristics ' + str(models_or_heauristics)
                )

            # Take only "beam_width" of the best states (i.e. most nearest to destination according to the model estimate)
            idx = torch.argsort(estimations_for_new_states)[:beam_width]
            array_of_states = array_of_states_new[idx, :]

        else:
            # If number of states is less than beam_width - we take them all:
            array_of_states = array_of_states_new

        if verbose >= 10:
            print(
                i_step, 'i_step', array_of_states_new.shape, 'array_of_states_new.shape'
            )

    dict_additional_data = {}
    if verbose >= 1:
        print()
        print('Search finished.', 'beam_width:', beam_width)
        if flag_found_destination:
            print(i_step, ' steps to destination state. Path found.')
        else:
            print('Path not found.')

    return flag_found_destination, i_step, dict_additional_data


def get_unique_states_2(states: torch.Tensor, vec_hasher: torch.Tensor) -> torch.Tensor:
    """
    Return matrix with unique rows for input matrix "states"
    I.e. duplicate rows are dropped.
    For fast implementation: we use hashing via scalar/dot product.
    Note: output order of rows is different from the original.
    """
    # Note: that implementation is 30 times faster than torch.unique(states, dim = 0) - because we use hashes  (see K.Khoruzhii: https://t.me/sberlogasci/10989/15920)
    # Note: torch.unique does not support returning of indices of unique element so we cannot use it
    # That is in contrast to numpy.unique which supports - set: return_index = True

    device = states.device

    t1 = time.time()
    # Hashing rows of states matrix:
    hashed = torch.sum(states * vec_hasher.to(device), dim=1)   # Compute hashes.
    # It is same as matrix product torch.matmul(hash_vec , states )
    # but pay attention: such code work with GPU for integers
    # While torch.matmul - does not work for GPU for integer data types,
    # since old GPU hardware (before 2020: P100, T4) does not support integer matrix multiplication
    t1 = time.time() - t1
    print(t1, 'hash')

    # Sort
    t1 = time.time()
    hashed_sorted, idx = torch.sort(hashed)
    t1 = time.time() - t1
    print(t1, 'sort')

    # Mask selects elements which are different from the consequite - that is unique elements (since vector is sorted on the previous step)
    t1 = time.time()
    mask = torch.concat(
        (
            torch.tensor([True], device=device),
            hashed_sorted[1:] - hashed_sorted[:-1] > 0,
        )
    )
    t1 = time.time() - t1
    print(t1, 'mask')
    return states[idx][mask]


def get_neighbors(states, moves):
    """
    Some torch magic to calculate all new states which can be obtained from states by moves
    """
    return torch.gather(
        states.unsqueeze(1).expand(states.size(0), moves.shape[0], states.size(1)),
        2,
        moves.unsqueeze(0).expand(states.size(0), moves.shape[0], states.size(1)),
    )


flag_found_destination, i_step, dict_additional_data = beam_search_permutations_torch(
    state_start=[1, 0], generators=[[1, 0]], verbose=1
)
flag_found_destination, i_step, dict_additional_data = beam_search_permutations_torch(
    state_start=[2, 1, 0], generators=[[1, 0, 2], [0, 2, 1]], verbose=1
)
