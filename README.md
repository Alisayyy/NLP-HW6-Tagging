# NLP-HW6-Tagging
# 1
## a
### i
before: P_day1(H) =  0.871</br>
after: P_day1(H) = 0.491</br>


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

before when there are two ice-creams at day one, the probability of day one is a hot day is high. the probability of day 2 being a hot day is: P(H|H)*P_day1(H) + P(H|C)*P_day1(C). As p(H|H) and p(H|C) stay the same in the given chart, changing the ice cream in day one results in lowering the probability of a hot day, such that 

P(HHH|133) = p(H|start)*p(1|H) * p(H|H)*p(3|H) * p(H|H)*p(3|H)
            = 0.5 * 0.1 * 
P(CHH|133) = p(C|start)*p(1|C) * p(H|C)*p(3|H) * p(H|H)*p(3|H)



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
