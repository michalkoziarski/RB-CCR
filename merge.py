import pandas as pd

from pathlib import Path
from tqdm import tqdm


OUTPUT_DIRECTORY = Path(__file__).parent / 'results'


if __name__ == '__main__':
    OUTPUT_DIRECTORY.mkdir(exist_ok=True, parents=True)

    for trial in ['preliminary_energy', 'final']:
        input_directory = Path(f'results_{trial}')

        dfs = []

        for path in tqdm(input_directory.iterdir(), desc=f'Trial: {trial}'):
            df = pd.read_csv(path)
            dfs.append(df)

        df = pd.concat(dfs)
        df.to_csv(OUTPUT_DIRECTORY / f'results_{trial}.csv', index=False)
