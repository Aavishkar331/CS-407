# LLM Prompt Appendix (Task 3)

The model design (2-2-1, nonlinear hidden activation, sigmoid output + BCE,
the validation criteria) was written **before** prompting (Task 2).

## Prompt used

> Generate minimal PyTorch code for the following model and dataset. Do not
> change the architecture or task. The dataset is the four XOR examples:
> (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0. The network is 2 inputs -> 2 hidden
> units -> 1 output, with a selectable hidden activation (sigmoid, tanh, or
> ReLU) and a sigmoid binary output trained with `BCEWithLogitsLoss` (logits,
> for numerical stability). Use random weight initialisation and full-batch
> gradient descent for a few thousand CPU steps. Set a random seed for
> reproducibility. After training, report the final loss, the four
> probabilities, the thresholded labels, and one first-layer parameter-gradient
> tensor. Explain each test in one sentence.

### Follow-up prompts

- *Symmetry:* "Add an option to initialise all weights to zero and, after
  training, print the two rows of the first-layer weight matrix."
- *Three-class (Task 5):* "Modify only the output/loss portion: replace the
  single logit with three logits and use softmax + multiclass cross-entropy for
  the classes 0=(0,0), 1=disagree, 2=(1,1). Print the class probabilities for
  all four inputs and verify each row sums to 1."

## Corrections / changes made before accepting the code

1. **Logits, not probabilities, into the loss.** Kept the network output as raw
   logits and used `BCEWithLogitsLoss` (not `sigmoid` then `BCELoss`), for
   numerical stability.
2. **Early-step gradient capture.** Recorded the first-layer gradient norm at an
   early step (step 5), since at convergence gradients are tiny and
   uninformative for the activation comparison.
3. **Explicit backprop reference.** Because PyTorch was unavailable at run time,
   wrote an equivalent NumPy version (`xor_numpy.py`) with the backprop chain
   spelled out (`dz, dW2, dh1, da1, dW1`) and validated it against the expected
   `p − y` logit gradient and the known XOR behaviour.
4. **Stable softmax.** In the three-class extension, subtracted `max(logits)`
   before exponentiating, and added the "+100" shift-invariance diagnostic.
