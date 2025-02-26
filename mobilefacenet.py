import equinox as eqx
import jax
import jax.numpy as jnp


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(eqx.nn.Sequential):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, *, key
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            [
                eqx.nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    use_bias=False,
                    key=key,
                ),
                eqx.nn.BatchNorm(out_planes, axis_name="batch"),
                eqx.nn.Lambda(jax.nn.relu6),
            ]
        )


class DepthwiseSeparableConv(eqx.Module):
    depthwise: eqx.nn.Conv2d
    pointwise: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    bn2: eqx.nn.BatchNorm

    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False, *, key):
        key1, key2 = jax.random.split(key, 2)
        self.depthwise = eqx.nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_planes,
            use_bias=bias,
            key=key1,
        )
        self.pointwise = eqx.nn.Conv2d(
            in_planes, out_planes, kernel_size=1, use_bias=bias, key=key2
        )
        self.bn1 = eqx.nn.BatchNorm(in_planes, axis_name="batch")
        self.bn2 = eqx.nn.BatchNorm(out_planes, axis_name="batch")

    def __call__(self, x, state):
        x = self.depthwise(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)

        x = self.pointwise(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)
        return x, state


class GDConv(eqx.Module):
    depthwise: eqx.nn.Conv2d
    bn: eqx.nn.BatchNorm

    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False, *, key):
        self.depthwise = eqx.nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_planes,
            use_bias=bias,
            key=key,
        )
        self.bn = eqx.nn.BatchNorm(in_planes, axis_name="bias")

    def __call__(self, x, state):
        x = self.depthwise(x)
        x, state = self.bn(x, state)
        return x, state


class InvertedResidual(eqx.nn.StatefulLayer):
    conv: eqx.nn.Sequential
    use_res_connect: bool

    def __init__(self, inp, oup, stride, expand_ratio, *, key):
        assert stride in [1, 2]

        key1, key2, key3 = jax.random.split(key, 3)
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, key=key1))
        layers.extend(
            [
                ConvBNReLU(
                    hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, key=key2
                ),
                eqx.nn.Conv2d(hidden_dim, oup, 1, 1, 0, use_bias=False, key=key3),
                eqx.nn.BatchNorm(oup, axis_name="batch"),
            ]
        )
        self.conv = eqx.nn.Sequential(layers)

    def __call__(self, x, state, key=None):
        cout, state = self.conv(x, state)
        if self.use_res_connect:
            return x + cout, state
        return cout, state


class MobileFaceNet(eqx.Module):
    conv1: ConvBNReLU
    conv2: ConvBNReLU
    conv3: eqx.nn.Conv2d
    dw_conv: DepthwiseSeparableConv
    gdconv: GDConv
    features: eqx.nn.Sequential
    bn: eqx.nn.BatchNorm

    def __init__(
        self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, *, key
    ):
        block = InvertedResidual
        input_channel = 64
        last_channel = 512

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 5, 2],
                [4, 128, 1, 2],
                [2, 128, 6, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting)
            )

        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        # building first layer
        # input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        self.conv1 = ConvBNReLU(3, input_channel, stride=2, key=key1)
        self.dw_conv = DepthwiseSeparableConv(
            in_planes=64, out_planes=64, kernel_size=3, padding=1, key=key2
        )
        features = []
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t, key=key3)
                )
                input_channel = output_channel

        self.conv2 = ConvBNReLU(input_channel, last_channel, kernel_size=1, key=key4)
        self.gdconv = GDConv(in_planes=512, out_planes=512, kernel_size=7, padding=0, key=key5)
        self.conv3 = eqx.nn.Conv2d(512, 128, kernel_size=1, key=key6)
        self.bn = eqx.nn.BatchNorm(128, axis_name="batch")
        self.features = eqx.nn.Sequential(features)

    def __call__(self, x, state):
        x, state = self.conv1(x, state)
        x, state = self.dw_conv(x, state)
        x, state = self.features(x, state)
        x, state = self.conv2(x, state)
        x, state = self.gdconv(x, state)
        x = self.conv3(x)
        x, state = self.bn(x, state)
        x = jnp.reshape(x, (x.shape[0], -1))
        return x, state


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    mkey, xkey = jax.random.split(rng)
    model, state = eqx.nn.make_with_state(MobileFaceNet)(key=mkey)
    inf = eqx.nn.inference_mode(model)
    x = jax.random.uniform(xkey, (3, 112, 112))
    y, _ = inf(x, state)
    print(y.shape)
