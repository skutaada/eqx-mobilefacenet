import argparse
import json

import jax
import equinox as eqx
import spu.utils.distributed as ppd

from mobilefacenet import MobileFaceNet

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf['nodes'], conf['devices'])

model, state = eqx.nn.make_with_state(MobileFaceNet)(key=jax.random.PRNGKey(0))
model = eqx.nn.inference_mode(model)
params, static = eqx.partition(model, eqx.is_inexact_array)

def apply_fn(params, x):
    model = eqx.combine(params, static)

if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    mkey, xkey = jax.random.split(rng)
    model, state = eqx.nn.make_with_state(MobileFaceNet)(key=mkey)
    inf = eqx.nn.inference_mode(model)

    x = jax.random.uniform(xkey, (3, 112, 112))

    x_enc = ppd.device("P1")(lambda x: x)(x)
    y_enc = ppd.device("SPU")(inf)(x_enc, state)
    y = ppd.get(y_enc)
    print(y.shape)
