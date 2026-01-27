import itertools
import sys


if __name__ == '__main__':
    assert len(sys.argv) == 2
    sampling_step_multiplier = int(sys.argv[1])
    if sampling_step_multiplier <= 1:
        raise ValueError('The sampling step multiplier must be 2 or larger.')
    print('feynmann_kac_chain_count,source_parallel_tempering_chain_count,source_parallel_tempering_update_count')
    for feynmann_kac_chain_count in range(1, sampling_step_multiplier + 1):
        for source_parallel_tempering_chain_count, source_parallel_tempering_update_count in itertools.product(range(1, feynmann_kac_chain_count), repeat=2):
            if feynmann_kac_chain_count == source_parallel_tempering_chain_count * (source_parallel_tempering_update_count + 1):
                print(f'{feynmann_kac_chain_count},{source_parallel_tempering_chain_count},{source_parallel_tempering_update_count}')
