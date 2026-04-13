# Explaining the Stuff
**NOTE:** Don't include in the submission


## 1. Reading the file
- used the `pyarrow` engine for memory efficiency and speed (uses multi-threading)

- PyArrow types are generally much more memory-efficient (especially for strings) and handle missing data (NA) much cleaner

## 2. Hardware Used
- check to see if there is a compatible GPU using `torch.cuda.is_available`

- use `device` to send the data and model to the gpu

## 3. Type Conversion
- `pandas` and `pyarrow` usually use `float64` data type by default, but Pytorch models are designed to natively run with 32-bit-floats

## 4. Use of Ordinal Encoder
- In PyTorch deep learning, integer IDs are usually passed into an Embedding Layer (specifically `torch.nn.Embedding`). Instead of hard-coding 1s and 0s, an Embedding Layer takes these simple integer IDs and learns its own complex, multi-dimensional mathematical representations for every part of speech during the training process. It is highly advanced and much more memory-efficient!

## 5. 