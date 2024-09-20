import numpy as np
from RNNCell import RNNCell

def test_weight_sharing():
    # Initialize dummy inputs and hidden states
    inp1 = np.random.randn(10, 1)  # input vector of size 10x1
    inp2 = np.random.randn(10, 1)  # another input vector of size 10x1
    hs = np.random.randn(5, 1)     # hidden state of size 5x1

    # Create the first RNNCell
    rnn1 = RNNCell(inp=inp1, hs=hs)
    
    # Perform forward pass on rnn1 to generate an output
    rnn1.forward()

    # Create the second RNNCell using rnn1 as the predecessor (should share weights)
    rnn2 = RNNCell(inp=inp2, predecessor=rnn1)

    # Perform forward pass on rnn2
    rnn2.forward()

    # Test if weights are shared between rnn1 and rnn2
    shared_wh = rnn1.wh is rnn2.wh
    shared_wi = rnn1.wi is rnn2.wi
    shared_bias = rnn1.bias is rnn2.bias

    # Print results
    print(f"Weight sharing for wh: {'Passed' if shared_wh else 'Failed'}")
    print(f"Weight sharing for wi: {'Passed' if shared_wi else 'Failed'}")
    print(f"Weight sharing for bias: {'Passed' if shared_bias else 'Failed'}")

# Run the test
test_weight_sharing()
