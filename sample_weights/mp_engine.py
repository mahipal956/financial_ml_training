"""
Multiprocessing Engine for Financial Machine Learning (AFML Chapter 20)
Supports cross-platform parallel execution with clean single-core fallback for Windows.
"""
import sys
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool


def linParts(numAtoms, numThreads):
    """
    Partition numAtoms into numThreads chunks linearly.
    """
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nestedParts(numAtoms, numThreads, upperTriang=False):
    """
    Partition numAtoms into numThreads chunks with unequal load
    for nested loops (e.g., upper triangular operations).
    """
    parts = [0]
    numThreads_ = min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + numAtoms * (numAtoms + 1.) / numThreads_)
        part = (-1 + part ** 0.5) / 2.0
        parts.append(int(np.floor(part)))
    return np.array(parts)


def mpPandasObj(func, pdObj, numThreads=1, mpBatches=1, linMols=True, **kargs):
    """
    Parallelize functions operating on pandas objects (Series / DataFrame).
    func: The callback function to execute. Its first argument must be 'molecule' (indices chunk).
    pdObj: A tuple (arg_name, pd_series/index/list) passed as the molecule to chunk.
    numThreads: Number of parallel worker threads / processes.
    """
    arg_name, molecule_data = pdObj
    if isinstance(molecule_data, (pd.Index, pd.Series)):
        atoms = molecule_data.values
    elif isinstance(molecule_data, (list, np.ndarray)):
        atoms = np.array(molecule_data)
    else:
        atoms = np.array(list(molecule_data))

    if numThreads is None or numThreads <= 0:
        numThreads = max(1, cpu_count() - 1)

    # If single thread or short atoms list, run sequentially
    if numThreads == 1 or len(atoms) <= 1:
        kargs[arg_name] = atoms
        return func(**kargs)

    # Chunk molecules
    if linMols:
        parts = linParts(len(atoms), numThreads * mpBatches)
    else:
        parts = nestedParts(len(atoms), numThreads * mpBatches)

    jobs = []
    for i in range(1, len(parts)):
        sub_atoms = atoms[parts[i - 1]:parts[i]]
        if len(sub_atoms) > 0:
            job = {arg_name: sub_atoms}
            job.update(kargs)
            jobs.append(job)

    # Execute via Pool if numThreads > 1, with sequential fallback for unpicklable closures
    out = []
    if numThreads > 1:
        try:
            with Pool(processes=numThreads) as pool:
                async_res = [pool.apply_async(func, kwds=job) for job in jobs]
                out = [r.get(timeout=30) for r in async_res]
        except Exception:
            # Fallback to sequential execution across jobs
            out = [func(**job) for job in jobs]
    else:
        out = [func(**job) for job in jobs]

    if len(out) == 0:
        return pd.Series(dtype=float)

    if isinstance(out[0], pd.DataFrame):
        df0 = pd.concat(out, axis=0)
    elif isinstance(out[0], pd.Series):
        df0 = pd.concat(out, axis=0)
    else:
        return out
    return df0
