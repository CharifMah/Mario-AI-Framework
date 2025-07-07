# models/mariogan_lsi/gan/test_keras.py
import numpy as np
from model_keras import build_generator, build_discriminator

def test_shapes():
    G = build_generator()
    D = build_discriminator()
    z = np.random.randn(4, 32).astype("float32")
    fake = G(z)
    print("G output:", fake.shape)   # (4,16,256,1)
    out = D(fake)
    print("D output:", out.shape)    # (4,1)

if __name__=="__main__":
    test_shapes()
