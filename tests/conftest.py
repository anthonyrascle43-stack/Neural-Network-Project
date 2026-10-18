# Load torch's OpenMP runtime before scikit-learn's: with both libomp copies (e.g. Anaconda on macOS),
# importing scikit-learn first can segfault later torch calls.
import torch  # noqa: F401
