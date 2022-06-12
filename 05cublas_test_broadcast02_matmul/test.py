import numpy as np

def _load_tensor(file):     
    with open(file, "rb") as f:
        binary_data = f.read()

    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."
    
    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)

    if dtype == 0:
        np_dtype = np.float32
    elif dtype == 1:
        np_dtype = np.float16
    else:
        assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"
        
    return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)


def load_tensor(file):
    if file.endswith("npz"):
        return np.load(file)['data']
    elif file.endswith("npy"):
        return np.load(file)
    else:
        return _load_tensor(file)

def test():
    p_tensor = load_tensor('p_tensor.npz')
    q_tensor = load_tensor('q_tensor.npz')
    out_tensor1 = load_tensor('out_tensor1.npz')
    out_tensor2 = load_tensor('out_tensor2.npz')
    out_tensor3 = load_tensor('out_tensor3.npz')

    
    out1 = q_tensor @ p_tensor
    out2 = q_tensor[0:1] @ p_tensor
    out3 = q_tensor[:, 0:1, ...] @ p_tensor
    print("q_tensor[0:1].shape p_tensor.shape", q_tensor[0:1].shape, p_tensor.shape)
    
    print(np.abs(out1 - out_tensor1).max())
    print(np.abs(out2 - out_tensor2).max())
    print(np.abs(out3 - out_tensor3).max())


if __name__ == "__main__":
    test()
