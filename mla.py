from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.datasets import mnist

  
class Block:
  def __init__(self, in_dims, dims, stride=1):
    super().__init__()

    self.conv1 = nn.Conv2d(
      in_dims, dims, kernel_size=3, stride=stride, padding=1, bias=False
    )
    self.bn1 = nn.BatchNorm(dims)

    self.conv2 = nn.Conv2d(
      dims, dims, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.bn2 = nn.BatchNorm(dims)

    self.downsample = []
    if stride != 1 or in_dims != dims:  # Adjust for mismatched dimensions
      self.downsample = [
        nn.Conv2d(in_dims, dims, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm(dims)
      ]

  def __call__(self, x: Tensor) -> Tensor:
    # Main path
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out))
    
    # Residual path (downsampling if necessary)
    residual = x
    for layer in self.downsample:
      residual = layer(residual)
    
    # Combine paths (no in-place operation)
    out = out + residual
    return out.relu()



class Model4:
    def __init__(self):
        
        self.conv = nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2)  
        self.relu = Tensor.relu

        # Dynamically generate 50 blocks
        self.blocks = self._generate_blocks(num_blocks=40)

        
        dummy_input = Tensor.zeros(1, 1, 28, 28)  
        output_tensor = self._forward_blocks(self.conv(self.relu(dummy_input)))
        flattened_size = output_tensor.shape[1] * output_tensor.shape[2] * output_tensor.shape[3]  # C × H × W

        # Final layers
        self.layers: List[Callable[[Tensor], Tensor]] = [
            self.conv, self.relu,  
            *self.blocks,          
            lambda x: x.flatten(1), 
            nn.Linear(flattened_size, 10)  
        ]

    def _generate_blocks(self, num_blocks):
        """Generates a list of Block instances with varying dimensions."""
        blocks = []
        in_channels = 4  # Start with 2 channels (matches initial conv output)
        for i in range(num_blocks):
            out_channels = in_channels + (1 if i % 10 == 0 else 0)  # Increment output every 10 layers
            blocks.append(Block(in_channels, out_channels, stride=1))
            in_channels = out_channels  # Update input channels for the next block
        return blocks

    def _forward_blocks(self, x):
        """Passes the dummy input through the blocks to calculate the output shape."""
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

  model = Model4()

  opt = nn.optim.Adam(nn.state.get_parameters(model))

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(getenv("BS", 32), high=X_train.shape[0])
    
    loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
    opt.step()
    return loss

  @TinyJit
  @Tensor.test()
  def get_test_acc() -> Tensor: return (model(X_test).argmax(axis=1) == Y_test).mean()*100

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 150))):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    loss = train_step()
    if i%10 == 9: test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if test_acc >= target and test_acc != 100.0: print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))