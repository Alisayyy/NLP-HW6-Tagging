# NLP-HW6-Tagging
# 1
## a
### i
before: P_day1(H) =  0.871</br>
after: P_day1(H) = 0.491</br>

From the forward perspective, the probability of C|Start and H|Start is equal, but the probability of 1|C is 0.7 and 1|H is 0.1. From the backward perspective, the second day is H. The probability of transition from H to H is 0.8 while the probability of transition from C to H is 0.1. Thus, timing them together, the probability that day 1 is hot is roughly 0.5.

### ii
before: P_day2(H) = 0.977</br>
after: P_day2(H) = 0.918</br>
This is record by the P(->H) on day 2 cell (k28).</br>

The change to day 1 data affects the forward probability of day 2 since day 1 is before day 2. For day 2, α(H) decreases and α(C) increases. This can be explained by the formular:

```
before: α(H) = P(C|start) P(2|C) * P(H|C) P(3|H) + P(H|start) P(2|H) * P(H|H) P(3|H)
```
```
after: α(H) = P(C|start) P(1|C) * P(H|C) P(3|H) + P(H|start) P(1|H) * P(H|H) P(3|H)
```
In our initial emission probability, P(1|C) > P(2|C) and P(1|H) < P(2|H), so the probability of paths to a C state at day 1 increase and the probability of paths to a H decreases. 

Furthermore, since the probability of trasition from H to H is higher than that of C to H (P(H|H) > P(H|C)). This means for the two sources of path to day 2 H, which is day 1 C and day 1 H, the latter one has more weight. Thus the overall α(H) at day 2 decreases.

Similar analysis for why α(C) increases.

Thus the probability of day 2 is hot decreases since it's calculated by 
```
P(->H) = α(H)β(H) / (α(H)β(H) + α(C)β(C))
```
and backward probability remain unchanged.

### iii
before: P_day1(H) = 1.0, P_day2(H) = 0.995</br>
after: P_day1(H) = 0.0, P_day2(H) = 0.557</br>

During the 10-iteration training, the probability of getting a hot day 1 gets smaller and smaller and finally reaches 0. The probability of day 2 being a hot day also gets smaller. Observed from the data, all other 1-ice-cream day happen relatively consecutively in the middle of the summer. From those data, the model learns that P(1|H) is 1.6E-04, which is extremely small, and tansitions between different states(H to C, C to H) are harder. Thus, the probability of day 1 with 1-ice-cream being hot gets very low and the probability of day 2 being hot also gets smaller since tranbsition from C to H gets harder.  


## b
### i
It will be certain about the days that eat only 1 ice cream. So when it is a one-ice-cream day, the weather assumption will return p(H) = 0 regardless of any other features or trend.

### ii
After 10 iterations, there's no big difference in the graph. Because each iteration will lower the probabiltiy of a hot day when eating 1 ice cream. So even if there's no sterotype, after 10 iterations, the probablity of P(1|H) is close to 0.

### iii
p(1|H) = 0 <br>
As the forward and backward passes are executed for each of the 2^33 paths, p(1|H) for each path is infleunced by the intial probabilities and the observed data. But each time there's only 1 ice cream and H, the probability of this path is to time p(1|H) which is 0.<br>
Forward pass: since p(1|H) = 0, any path involving icecream 1 in state H will have alpha = 0<br>
Backward pass: also 0 for the same reason<br>
Since p(1∣H)=0 is enforced, there is no opportunity for the model to adjust the emission probability for 1 in state H. In each iteration, the biased emission probabilites persist, both forward and backward passes continue to assign 0 probability to paths involving emission 1 in state H.


## c
### i
The state corresponding to the initial/start state will have a β probability equal to the total probability of the sentence. This is because the β probability for the initial state captures the probability of observing the entire sequence starting from the initial state.

### ii
H constituent represents a state/tag in the HMM. It likely corresponds to a specific state in the trellis used for tagging sequences. <br>
The probability of the rule H -> 1 C is the likelihood of transitioning from state H to state C and emitting 1 (a transition from a hot day to a cold day after eating one ice cream). P(1|C)*P(C|H) <br>
H->ε represents the transition from a hot day to stop. P(STOP|H) <br>
Reason:
- Capture more complex relationships: The more complicated parse on the right can be more expressive in modeling the underlying patterns in the sequence, so for example it can represent the non-terminals.
- Handle ambiguities or variability


# 2
## a
$α_{BOS}(0)$ is the forward probability of being in the start state, $β_{EOS}(n)$ is the starting point for backward algorithm. By setting the start point to 1, the sequence have start and stop state both equals 1 which represent the total probability of the sequence to be 1. The algorithm initalizes the probaility of starting the sequence, so that the distribution is normalized.

## b
Perplexity represents how well a model predicts a sample. When training on sup file and evaluate on a held-out raw file, the model has learned from the supervised training data. And as the raw file shares similar patterns, the model can predict the correct patterns. <br>
But when evulate on held-out dev files, the model finds its own way to learn the pattern from these diverse set of training data. Therefore, it might deviate in some way from the distribution seen during training. The higher perplexity suggest that the model struggles more to predict the held-out dev set. <br>
The perplexity on a held-out raw file is more important because it represents how will the model perform on real-world data.

## c
Because dev is used for evaluation of the model. And if V includes word types from dev, the model could potentially learn specific features unique to the dev set, leading to overfitting and reduced generalization to new data which will have biased evaluation. The model should be able to generalize the knowledge learned from training data to perform on unseen set of data.

## d


## e
only on sup:<br>
Tagging accuracy: all: 93.044%, known: 95.449%, seen: 72.222%, novel: 66.513%

semi-supervised:<br>
Tagging accuracy: all: 92.856%, known: 95.252%, seen: 70.875%, novel: 66.909%

It hurts the overall accuracy but improve on novel accuracy. The model might be specializing too much in capturing the novel instances, which might lead to a decrease in accuracy on seen instances.

## f
The semi-supervised training will sample a tag to the word that the model thinks has a high chance of being correct. And this training will give the model more tags which will make the problem easier. So that even we never observe a tag associate with a word in the raw data, the model might be able to guess this tag based on observing the sentence structure in an untagged sentence given the bigram information. 

For example, the model knows nothing about "cavier", but in raw data there exists "the cavier with" and the model knows about the tagging for "the/Det" and "with/Prep". Then it will want to guess "cavier/Noun" in between because it knows the tag bigrams "Det Noun" and "Noun Prep".

## g
1. If there's too much unseen words in a sentence, the model might not be able to generate useful information given the known tags. When most of the words are unseen, the model is unable to guess from the a minimal number of words with known tags.
2. The distribution of data in the supervised dataset differs significantly from the distribution of the raw dataset. If the model learns from the labeled data and generalizes its knowledge to the raw data, but the raw data's distribution is substantially different, the model may not gain useful information from raw data and could possibly be misled.

## h
enraw:<br>
Tagging accuracy: all: 8.410%, known: 5.201%, seen: 3.030%, novel: 56.803% <br>
Tagging accuracy is quite low across all categories, indicating that relying solely on raw data without any supervision leads to poor performance. This is expected, as raw data may be noisy and lacks labeled examples for the model to learn from.

ensup + enraw:<br>
Tagging accuracy: all: 92.856%, known: 95.252%, seen: 70.875%, novel: 66.909%<br>
The supervised data (ensup) helps the model learn more accurate representations, and combining it with raw data helps generalize to unseen examples.

ensup + enraw + ensup: <br>
Tagging accuracy: all: 93.077%, known: 95.458%, seen: 72.391%, novel: 66.843%

ensup + ensup + ensup + enraw: <br>
Tagging accuracy: all: 93.094%, known: 95.476%, seen: 72.391%, novel: 66.843%<br>
Weighting the supervised data more heavily by repeating it multiple times in training seems to be beneficial, possibly because it provides more reliable signals for learning..


# 3
## a
For awesome, we implement "constraints on inference". We add a tag dictionary tensor for every word type to record all tags appear for this word in the supervised training data. The tensors are initializaed to be 0 (Actually we initialize the values to be 1e-45 instead of 0 to avoid log(0)=-inf in later process). Everytime a tag occurs for this word, its corresponding position in tensor will change to 1. Thus, later when timing this tensor, the probability of tags never appear for the word will have close to zero prabability.

For OOV cases, during viterbi, we first calculate a smoothed probability based on the dev data and update "self.B[:,corpus.integerize_word("_OOV_")]". This is becasue during the training process, parameters in self.B regarding OOV will not learn at all. We also don't want to assign zero probability or equal proability for every tag to OOV words. Thus, we count the frequency of each tag for OOV word and smooth it.

## b
We train on both ensup and enraw. The one with --awesome tag has:

Tagging accuracy: all: 92.856%, known: 95.252%, seen: 70.875%, novel: 66.909%

and the one without --awesome tag reports:

Tagging accuracy: all: 80.216%, known: 81.635%, seen: 72.391%, novel: 62.814%

The overall, known words, and novel words accuracy have noticable increase. For known words, the improvement is the most significant because the tag dictionary excludes all impossible tags (tags never appear in sup data) for a word when tagging it. The smoothed probability gives better prediction of tags for novel words that never appear in sup or raw.

However, the seen words accuracy decreases with awesome tag. This is because those words all appear with no tag during training so the tag dictionary assigns equal probability to all of them.


# 4
## a
CRF ensup:<br>
Tagging accuracy: all: 93.231%, known: 95.669%, seen: 74.411%, novel: 65.456%

HMM ensup: <br>
Tagging accuracy: all: 93.044%, known: 95.449%, seen: 72.222%, novel: 66.513%

Both model performs well on known words. CRF outperforms HMM on seen words by a small margin, but HMM has a slightly better accuracy on novel words compared to CRF.

Error patterns: Both models make errors in predicting the token "OOV." Sometimes CRF predicts it as a noun, while HMM predicts it as a verb. This suggests a challenge in handling out-of-vocabulary tokens.<br>
The HMM model tends to predict "OOV" as a noun (N), while the CRF model varies between predicting it as a conjunction (C) and a noun (N). These differences highlight the nuanced differences in how each model handles unseen or out-of-vocabulary words.

## b
CRF ensup:<br>
Tagging accuracy: all: 93.231%, known: 95.669%, seen: 74.411%, novel: 65.456%

CRF ensup+enraw:<br>
Tagging accuracy: all: 93.244%, known: 95.687%, seen: 74.242%, novel: 65.456%

enraw has no effect on improving the accuracy of the model. The differences in tagging accuracy between the two models seem relatively small. In general, the addition of "enraw" does not appear to have a significant impact on the overall performance of the model, as the accuracy values are similar between the two cases.

This is because CRF is a discriminative model that  primarily influenced by the labeled training data and focus on learning the relationships between input features (words) and output labels (tags).

## c
We didn't implemement the biRNN-CRF.

But we assume that it should perform better than the basic CRF because the biRNN-CRF is designed to capture sequential patterns and dependencies. With tag labeling that depends heavily on the the word sequence, biRNN-CRF should perform well in cases where the relationships between neighboring elements are sufficient for accurate labeling.