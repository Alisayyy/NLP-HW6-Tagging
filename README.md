# NLP-HW6-Tagging
# 1
## a
### i
p(H|Start)*p(1|H) 0.5*0.1 = 0.05

before: α(H) * β(H)/(α(H) * β(H) + α(C) * β(C)) = 0.871
after: α(H) * β(H)/(α(H) * β(H) + α(C) * β(C)) = 0.491

before
after: P(HHH|133) = p_day1(H) * day2_p(H) * day3_p(H) = 0.491*0.918*0.975=0.439



### ii
before: 0.1*(p(H|H)*p(3|H)+p(H|C)*p(3|H)) = 0.1*(0.8*0.7+0.1*0.7)=0.0063
after: 0.05*(p(H|H)*p(3|H)+p(H|C)*p(3|H)) = 0.05*(0.8*0.7+0.1*0.7)=0.0035



before: 0.918
after: 0.977

p(->H) on day2


### ii
before 233: p_1(H) = 1, p_2(H) = 0.995
after 133: p_1(H) = 0, p_2(H) = 0.557

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
