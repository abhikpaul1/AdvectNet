# Confirmed Architecture

The current confirmed model is:

```text
input frames: X0 = t, X3 = t+30
        |
        v
frozen pretrained RAFT
        |
        v
forward/backward endpoint flows F0->3, F3->0
        |
        v
Super-SloMo intermediate flow approximation
        |
        v
backward-warp X0 and X3 toward alpha = 1/3 or 2/3
        |
        v
time-conditioned U-Net predicts blend mask + residual
        |
        v
generated t+10 or t+20 frame
```

## Trainable Parameters

Confirmed trainable component:

```text
U-Net only
```

Confirmed frozen component:

```text
RAFT optical-flow backbone
```

RAFT fine-tuning is experimental in the repository and is not part of the confirmed final architecture unless its validation metrics are explicitly reported.

## Training Objective

The confirmed model trains the U-Net with reconstruction, edge, and physics-guided regularization losses while RAFT remains frozen.

For a target intermediate frame $Y_\alpha$ at temporal fraction $\alpha \in \{1/3, 2/3\}$, the prediction is $\hat{Y}_\alpha$. The total loss is:

$$
L =
\lambda_r L_{\text{rec}}
+ \lambda_g L_{\text{grad}}
+ \lambda_a L_{\text{adv}}
+ \lambda_s L_{\text{src}}
$$

where:

$$
L_{\text{rec}}
=
\frac{1}{N}
\sum_i
\sqrt{(\hat{Y}_{\alpha,i} - Y_{\alpha,i})^2 + \epsilon^2}
$$

is the Charbonnier reconstruction loss.

$$
L_{\text{grad}}
=
\left\lVert |\nabla_x \hat{Y}_\alpha| - |\nabla_x Y_\alpha| \right\rVert_1
+
\left\lVert |\nabla_y \hat{Y}_\alpha| - |\nabla_y Y_\alpha| \right\rVert_1
$$

preserves cloud-edge structure.

The physics-guided advection consistency term is:

$$
L_{\text{adv}}
=
\frac{1}{N}
\sum_i
\sqrt{
\left(
(\hat{Y}_{\alpha,i} - X_{0,i})
+ F_{t \rightarrow 0,x,i}\frac{\partial \hat{Y}_{\alpha,i}}{\partial x}
+ F_{t \rightarrow 0,y,i}\frac{\partial \hat{Y}_{\alpha,i}}{\partial y}
\right)^2
+ \epsilon^2
}
$$

This is a practical regularizer inspired by scalar advection:

$$
\frac{\partial \theta}{\partial t} + \mathbf{u}\cdot\nabla\theta = S
$$

Here $F_{t \rightarrow 0}$ is the intermediate backward-warp flow estimated from frozen RAFT endpoint flows. It is not claimed to be a true atmospheric wind field; it is used as a motion-consistency prior.

The sparse source prior is:

$$
L_{\text{src}} = \lVert R_\alpha \rVert_1
$$

where $R_\alpha$ is the U-Net residual head. This discourages the residual branch from hallucinating the whole frame and keeps most changes explained by warping/advection.

Typical weights used in the prototype:

```text
lambda_rec  = 1.0
lambda_grad = 0.5
lambda_adv  = 0.15
lambda_src  = 0.05
```

## Validation Target

Himawari-9 provides 10-minute cadence, so validation uses actual intermediate frames:

```text
X0 = Himawari t
Y1 = actual Himawari t+10
Y2 = actual Himawari t+20
X3 = Himawari t+30
```

Model predictions are compared as:

```text
model(X0, X3, alpha=1/3) -> Y1
model(X0, X3, alpha=2/3) -> Y2
```
