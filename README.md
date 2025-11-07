## In this notebook I'm looking at addition within the attention layers.

In particular, a network needs to be able to add numbers with large number of digits, when its vocabulary only contains the individual digits (or collections of subdigits). Here I use two stacked attention layers

In this notebook we test it by having a vocabulary just of digits (1-9), and summing three-digit numbers. Thus it has to learn to carry over. Right now the minimum value of the three digit numbers is 500 and the max is 999, thus, every sum creates a 4 digit number.

Examples in the test set are not contained in the train set.

Other things i implemented, but couldnt get something properly training...
- min_val being between 100-500, then the result couyld actually be three digits
- min_val being a 1 or 2 digit number as this is a more interesting case where the "+" and "=" signs are in different places





As a side note, it would be interesting to see what token GPT2 uses to sum (100 + 101). I.e, does it use the '1' and '0' tokens, or does it also use the '10' token?