## Visualizing the inner world of a large language model (Llama 2 7B)

— Last update: 10/19/2023

This is a work in progress.  Comments or suggestions are welcome!  For longer messages, you can reach me at lawray.ai@gmail.com.

Comic courtesy of <a href = "https://www.smbc-comics.com/comic/conscious-6" target = "_blank" rel = "noreferrer noopener">SMBC</a>:  
<img src="/docs/assets/img/SMBC_LLM_consciousness.png" target = "_blank" rel = "noreferrer noopener" alt = "SMBC Sept. 19 2023" width="500"/>

It’s always been my philosophy that the best way to learn a model deeply is to hop into the code and track
variables through a few key scenarios.  Unfortunately, large language models (LLM) these days have billions
of parameters, so it’s difficult to develop a close feel for their internal structure at that level.

Still, there’s a lot of structure to the matrix values, even at a quick glance.  This post will give a tour of
select matrices within the 7 billion parameter Llama-2 model and explore some of what the model has in mind as
it generates text.  Llama-2 is a popular LLM that was released by Meta on July 21, 2023, with <a href = "https://arxiv.org/abs/2307.09288" target = "_blank" rel = "noreferrer noopener">this accompanying paper</a>.  Some related learning resources are listed at the end of the article.

I’ll devote a short section to each of these topics:
1. What are the matrices, and how do they add up to 7B parameters?
2. What can we directly decode from internal states of the model?  
    2.A. Word association in the token encoding vector spaces  
    2.B. Internal dictionaries of an LLM
3. What do the first attention heads look for?
4. How do deep and shallow layers differ?
5. What do the layer outputs look like?
6. Lessons for LLM architecture
7. Useful links

### 1. What are the matrices, and how do they add up to 7B parameters?

<img src="/docs/assets/img/Llama-transformer.png" target = "_blank" rel = "noreferrer noopener" alt = "Llama transformer diagram" width="250"/> &nbsp;&nbsp; <img src="/docs/assets/img/llama-attn-diagram.png" target = "_blank" rel = "noreferrer noopener" alt = "Llama attention diagram" width="450"/>
    
**Figure 1: (left) A Llama-2 transformer block, and (right) a Llama-2 7B attention block.**

#### The full set of matrices for Llama-2 is <a href = "/docs/Llama-2/Llama-2-tensors.md" target = "_blank" rel = "noreferrer noopener">listed here</a>.  Let’s tally the parameters: 
Input encoder and output layers that map between the model dimension (4096) and the token vocabulary (32000).  I’ll refer to both of these 32000x4096 matrices as token ‘dictionaries’ in the text below.  **That’s 2 * 4096 * 32000 = 262,144,000 parameters.**
1. Weight matrices for the transformer attention mechanism (Wk,Wq,Wv,Wo).  These are stored as 4096x4096 tensors in the Llama-2 download, but should be thought of as 32x128x4096 (or permutations thereof), as there are 32 attention heads.  **That’s 4 * 4096<sup>2</sup> = 67,108,864 attention parameters per transformer.**
2. It’s worth noting that these head-specific matrices act in pairs as Wv<sub>h</sub><sup>T</sup>Wo<sub>h</sub> and Wq<sub>h</sub><sup>T</sup>R<sup>T</sup>RWk<sub>h</sub>, where the R matrices apply positional encoding. Each Wq<sub>h</sub><sup>T</sup>R<sup>T</sup>RWk<sub>h</sub> pair behaves very similarly to the <a href = "https://arxiv.org/abs/2106.09685" target = "_blank" rel = "noreferrer noopener">LoRA representation</a> of a larger 4096x4096 tensor.
3. Weight matrices for the feed forward network, which maps from the model dimension (4096) to a higher internal dimension (11008), and then back to the model dimension.  This would typically involve two linear layers (4096x11008 and 11008x4096), but Llama uses a SwiGLU activation for the first layer which requires an additional matrix.  **That’s 3 * 4096 * 11008 = 135,266,304 feed forward parameters per transformer.**
4. Note that there are 32 transformer layers, so one has 32 inequivalent versions of the matrices described in points (2-3)!  Each layer also includes two 4096-long rescaling vectors within the Llama equivalent of batch normalization (RMSNorm), and I see one final rescaling operation on the last transformer output - I’ll touch on this in Section 5. There is also a vector containing 64 frequencies used to create the <a href = "https://arxiv.org/abs/2104.09864" target = "_blank" rel = "noreferrer noopener">relative position encoding</a> R matrices. Putting it all together, we get:
<pre>
  262,144,000 token dict + 64 position encoding freq + 4096 final RMSNorm   
  32 layers * (67,108,864 attention + 135,266,304 feed forward + 2 * 4096 RMSNorm) 
= <b>6,738,415,680 total parameters</b>
</pre>
#### We’ll also take a look at several internal state matrices:
1. Attention matrices (#tokens x #tokens): Each attention head creates its own attention matrix, so if a context window contains 1000 input tokens, there will be 32 heads x 1000x1000 matrices per transformer layer.  However, with masked attention, only the bottom row of these attention matrices (a 1000-long vector) remains relevant for next-token generation.
2. The attention block outputs #tokens x 4096 state vectors that are added to the transformer inputs - this recombination with earlier states is called a persistent connection (see curved lines in Fig. 1, left).  These vectors are interesting, but do not need to be cached when running the model, and are not explored closely in the current version of this document.
3. After the attention block persistent connection, the #tokens x 4096 state vectors are fed into the feedforward layer, which acts on each 4096-long state vector independently - it is not synthesizing information between the different state vectors.  The feedforward layer outputs another set of #tokens x 4096 state vectors, which we’ll look at in Sections 2 and 5.  Only one of these 4096-long vectors needs to be created when generating a new token, however the previous vectors all need to be cached. Counting the first layer’s input as a 33rd ‘layer’, this suggests a minimum **cache size of 270 MB for a prompt size of 1000**. (270 MB = 33 layers x #tokens x 4096 x 2 bytes/float)

### 2. What can we directly decode from internal states of the model?

In the subsections below, we’ll see that some of what the model is ‘thinking’ can be decoded from three separate sets of word-to-vector mappings.  Information is passed forward through the model in the form of N sets of 4096-long vectors, where N is the number of input tokens.  The use of masked attention in Llama-2 means that these vectors are computed sequentially.  The N-1’th output of each layer is fully created before the model begins to work on the layer corresponding to the N’th token, which has access to all earlier information.

At the start and end of the model, the 32000x4096 **input encoder** and **output** layers give a direct mapping between these internal 4096-long states and specific tokens (pieces of words).  I’ll refer to these matrices as ***dictionaries***, because they allow us to translate between the model activations and english words.  A third dictionary (and I suspect there are more) is only found in layer outputs within the transformer stack, and seems to reveal word associations that help the model parse meaning.  I’ll refer to this as the **‘middle’ dictionary**.

The input and output dictionaries have significantly correlated internal structures (<a href = "/docs/input-vs-output.md" target = "_blank" rel = "noreferrer noopener">analysis here</a>), with meaningful relationships that I’ll circle back to in Figure 4.  However, all three of the dictionaries are approximately orthogonal to one another, and we’ll see that the model can use them simultaneously to encode different ‘registers of thought’.  The average Pearson correlation coefficient between input and output vectors for the same token is 0.016, just as for vectors of normally distributed random numbers (0.016 ~ 1/sqrt(4096)).

The vectorization of tokens in an LLM can bake in a lot of meaning, such as the famous **v_king - v_man + v_woman ≈ v_queen** relation between <a href = "https://carl-allen.github.io/nlp/2019/07/01/explaining-analogies-explained.html" target = "_blank" rel = "noreferrer noopener">token embedding vectors in word2vec</a>. I haven’t been able to reproduce that sort of equation for Llama-2 7B, and there is a great reason to expect it not to hold: the final (and ‘middle’) state vectors of the model represent superpositions of dissimilar words, and an additive logic within the closely related vector spaces – say, **v_woman + v_crown = v_queen** – would create problems for this. If the model wanted to consider both “woman” and “crown” as possible next-word candidates, it would end up outputting “queen” instead.

So, let’s look at a few word encodings!

#### 2.A. Word association in the token encoding vector spaces:

Let’s take a closer look at just the input dictionary, which has a <a href = "/docs/input-vs-output.md" target = "_blank" rel = "noreferrer noopener">similar structure</a> to the output token dictionary.  First, some pictures:

<img src="/docs/assets/img/input-dict-corr-girl.png" target = "_blank" rel = "noreferrer noopener" alt = "Input dict vector correlations" width="800"/>

**Figure 2: Similarity of input word encodings.** Normalized inner products between the 4096-long encoding vectors for different single-token words. (amplitude A = <v1|v2> / sqrt(<v1|v1><v2|v2>), for vectors v1 and v2 read from the 32000x4096 input encoder layer)

One thing that jumps out is that words with similar meanings are fairly consistent in having large inner products.  More than that, part of speech is a factor – the adjectives in Fig. 2 (“round”, “sharp”, “long”, and “short”) also have greater than average inner products with one another, and with other adjectives I’ve tested.  Adjectives addressing a similar property (“long”/“short”, “male”/”female” “red”/”yellow”) have even closer encodings.

These correlations probably serve several purposes.  On the one hand, they allow a more compact (lower effective rank) representation of the dictionary, which will reduce accidental spillover (noise) onto other information encoded in the same 4096-long state vector.  On the other, they make it easier for the key and query (Wk and Wq) matrices of attention heads to look for part of speech agreement or for other specific characteristics (say, colors).

For the output, this structure means that similar words will reinforce one another.  If you use the output dictionary to create an output vector containing an equal superposition of 3 tokens and two of them are colors, the chance of the model generating a color as the next token will be much greater than 2/3!  For example, defining **v = v_color1 + v_color2 + v_other** will give output (logit) values **<v\|v_color1> = <v\|v_color1> > <v\|v_other>**, for L2 normalized single-token vectors.

It’s also noteworthy that the scale bar of Fig. 2 has only positive numbers. This wasn’t deliberate, and it turns out that token inner products are positive 81% of the time. My sense from casual examination is that they're generally (but not universally) positive for pairings of English words or word-parts, and more often negative for English-to-foreign pairings (particularly with East Asian characters). This seems to suggest that words from different languages can behave competitively, suppressing one another’s amplitude in the output dictionary register.

There’s an even subtler form of word association that we can’t see in Fig. 2 – *words* in the input dictionary tend to look like *word completions* in the output dictionary.  For example, the input embedding of “play” has large inner products with “ground” and “offs” in the output encoding!  To get into this, we’ll want to start looking at internal layers of the model.

#### 2.B. Internal dictionaries of an LLM:

The persistent connections in the model (curved lines in Fig. 1(left)) cause the inputs to any given transformer layer to be partially copied over into the outputs, suggesting that the model will continue to have a relationship with these encodings over multiple transformer layers.  So… what happens when we use the input and output dictionaries to interpret layer outputs that they would usually not have access to?

Let’s consider a short prompt containing a few clear internal word relationships: “**I like the red ball, but my girlfriend likes the larger ball. We like to go to the basketball court to play ball together. It is a lot of fun.**”

If we use the model’s input dictionary to translate internal transformer layer outputs, we get plots like these:

<img src="/docs/assets/img/Input_dict_like.png" target = "_blank" rel = "noreferrer noopener" alt = "Input dict word amplitudes" width="500"/>
<img src="/docs/assets/img/Input_dict_ball.png" target = "_blank" rel = "noreferrer noopener" alt = "Input dict word amplitudes" width="500"/>

**Figure 3: Reading internal states using the input encoding.** Transformer output vectors (4096-long vectors) in the (top) 3rd word token=“like” position and (bottom) 6th word token=“ball” position are decoded using the 32000x4096 input encoding matrix. Amplitude of the five largest vector elements is plotted and overlaid with the words that each matrix element represents. The transformer outputs are L2 normalized, but the input encoding matrix is not.

The prompt contains 37 tokens, so the output of each transformer is a set of 37 4096-long vectors corresponding to each of the input token positions:

0: “\<s>”  
1: “I”  
2: “like”  
3: “the”  

… and so on, where “\<s>” is a dummy token placed at the start of every prompt. Projecting these vectors onto the input dictionary reveals that each input word continues to be readable all the way through the network.  There can be a little drift, like conflating “ball” and “Ball”, but it’s striking that this piece of information is protected from exponential decay as it propagates through 30 transformer layers. Instead, the amplitude drops to a roughly constant level beyond the 10th layer (A ~ 0.15 to 0.2), with a weight that corresponds to A<sup>2</sup>~ 3% of the full state vector.

Thinks get more interested in the output encoding:

<img src="/docs/assets/img/Output_dict_to.png" target = "_blank" rel = "noreferrer noopener" alt = "Output dict word amplitudes" width="600"/>

**Figure 4: Reading internal states using the output dictionary.** Output works poorly with short prompts, so I’ve chosen the 25th token position to decode (“to” in “basketball court **to** play ball”).

Unlike the projection onto input encoding, this output projection shows us collections of words that aren’t all that closely associated, such as “shoot” and “play”. My favorite thing about this plot is the set of words in “layer 0”, which shows how the raw input encoding of “to” is read in the output encoding language.  Rather than being meaningless, the “to” input vector projects strongly onto a set of tokens that can expand the word into “tops”, “topped”, “topping”, and “toast”!  The input and output encodings don’t just have similar structures – they’re also nestled with one another in a way that causes the input embeddings to automatically suggest sequential tokens when read with the output encoding.

In practice, these layer 0 continuations are usually incorrect, plots like Fig. 4 stop showing them within the first few layers.  The pattern I tend to see is that plausible words start to appear around the middle of the model (layers 10-20), and the word set becomes much better informed by logic and context in the last ~10 transformer layers.  A recent paper found that most factual knowledge seems to be encoded in the <a href = "https://arxiv.org/abs/2310.02207" target = "_blank" rel = "noreferrer noopener">first half of the model</a>, so it may be that the second half of the model is specialized in logical processing.

Here’s another example of output-based decoding that shows similar patterns:

<img src="/docs/assets/img/Output_dict_play.png" target = "_blank" rel = "noreferrer noopener" alt = "Output dict word amplitudes" width="600"/>

**Figure 5: Reading internal states using the output dictionary.** The input word is “play”, as in “basketball court to **play** ball”).

The decodings of the internal states in Fig. 3-5 are still missing something important.  We can see that the model remembers its prompts, and we can see it spitballing ideas for its output, but we can’t see the kind of pairwise word associations that one would expect from the key/query structure of attention.  To look a little bit further, let’s create a very approximate dictionary based on how the model transforms tokens in its first few layers (<a href = "/docs/Llama-2/Creating-Middle.md" target = "_blank" rel = "noreferrer noopener">method here</a>).  I’ll refer to this as the **“middle” dictionary**.

Here’s what happens when we try to use it…

<img src="/docs/assets/img/middle-I-like.png" target = "_blank" rel = "noreferrer noopener" alt = "Middle dict word amplitudes" width="600"/>

**Figure 5: Reading internal states using the middle dictionary.** The 3rd token position is decode (“like” in “\<s> I **like** the red ball”).

Several things stand out:
1. The middle dictionary has little overlap with the input and output dictionaries, which gives us very weak amplitudes in layers 0 and 32.  This is not an entirely trivial observation – it says that the model has opened up a near-orthogonal sector of its 4096-dimensional state space.
2. Unlike the input and output dictionaries, the middle dictionary shows us the previous word in the sentence - “I”!  Due to the attention masking within Llama and GPT-family models, the layer outputs in the 3rd token position (“\<s> I **like**”) are only aware of the input tokens “I”, “like”, and the dummy token “\<s>”.  I’ve removed \<s> from the middle dictionary basis by hand, and the remaining words “I” and “like” are both visible in this middle encoding register.
3. The middle encoding vectors are probably rather inaccurate.  Words that are modestly related to “I” show up with similar amplitudes, including “We”, “my”, and fellow pronoun, “it”.

To mitigate the inaccuracy of the middle dictionary, we can limit the basis to just the 28 unique tokens that exist in the prompt. Here’s what that looks like for the 6th token position:

<img src="/docs/assets/img/middle-red-ball.png" target = "_blank" rel = "noreferrer noopener" alt = "Middle dict word amplitudes" width="600"/>

**Figure 6: Reading internal states using the middle encoding.** The decoded token position corresponds with “ball” (“\<s> I like the red **ball**”).  The top eight word matches are plotted.

The words “I”, “red” and “ball” are near the top of the list, possibly labeling the ball as a “red ball”.  However, the confusion between “I”, “We”, “my” and “it” fills up most of the top of the word register, making the plot difficult to read.

The real problem with this dictionary is that we don’t know what the model is using it for or how similar it really is to the input and output dictionaries.  I would assume that the ‘words’ include relational information between different words as well as other nuances that we’re missing in this brute force translation.  There are more <a href = "https://arxiv.org/abs/1610.01644" target = "_blank" rel = "noreferrer noopener">computationally intensive approaches</a> that could be used to try to dig this out, but let’s move on for now.

### 3. What words do the attention heads look for?

The four weight matrices (Wk,Wq,Wv,Wo) are each stored as 4096x4096 tensors in the Llama-2 download, but are partitioned into 32 128x4096 “attention head” tensors (Wkh, Wqh, ...).  A heatmap representation of the matrix elements for one of these matrices <a href = "/docs/Llama-2/Weight-matrix-heatmap.md" target = "_blank" rel = "noreferrer noopener">can be viewed here</a>, together with a few notes on the features visible by eye.

**One starting point** to dissect these matrices is to think of the 4096-long vectors as representing words, as they function through inner products with the token encodings. Taking an inner product between input token encodings and key/query vectors in the first layer reveals sensible patterns for the individual attention heads, with the ‘key’ vectors looking for tokens that occur to the left of the ‘query’ vectors.

The very first head is a great example.  For head [0] (<a href = "/docs/Llama-2/Head0-layer0-word-mappings.md" target = "_blank" rel = "noreferrer noopener">code output here</a>), the key vectors map strongly to open parentheses and words that frequently open a parenthetical note such as "although", "occasionally", "approximately", “File”. This head also looks for word endings such as “ingly”, “demic”, and “aneous” that have similar associations – think “(amazingly, …”, “(surprisingly, …”, “(simultaneously, …” and so on.  The corresponding 0-indexed head query features closing parentheses, in keeping with the ‘key tokens to the left, query tokens to the right’ principle.  Head [5] does something similar for quotations.

On the other hand, it’s overly simplistic to think of the head vectors as searching for single words.  And we’re glossing over the positional encoding.  As things stand, it’s unusual to see a single word with more than a ~30% projection onto a given 4096-long vector within Wk and Wq, so if we think of these vectors as words, we need to at least think of them as superpositions of words.

**In that vein, an alternative starting point** based on the correlations in Fig. 2 might be to consider that the encoding vector spaces may contain clustered vectors associated with similar properties or parts of speech. The key/query vectors could be pointing to the centers of these clusters – for example, to look for related adjective/noun pairs.

It should also be noted that even though Llama-2 is a language model, the information encoded in model parameters is not purely linguistic. For example, a fascinating <a href = "https://arxiv.org/abs/2310.02207" target = "_blank" rel = "noreferrer noopener">recent paper</a> showed that there are neuron activations inside the Llama models that may give it a sense of continuous dimensions such as time and latitude/longitude.

### 4. How do deep and shallow layers differ?

The distribution of Wk/Wq/Wv/Wo values is approximately 0-centered, and evolves from a fat-tailed distribution towards a gaussian distribution as one goes deeper in the network.  A great metric to track this trend with is the ratio of standard deviation to the mean amplitude [σ/mean(abs(vect))], which has a value of ~1.25 for a 0-centered gaussian.

<img src="/docs/assets/img/sigma-over-mean.png" target = "_blank" rel = "noreferrer noopener" alt = "Sigma divided by mean for attention matrices" width="600"/>

**Figure 6: Parameter distribution by layer.**  Tracking the metric σ/mean(abs(vect)) versus layer in the neural network for 4096x4096 representations of the Wk, Wq, Wv, and Wo attention matrices. Blue curves show the mean value for vectors oriented along the 4096-long model dimension, while orange curves show the mean value for 4096-long vectors that cut across the attention heads. 

Parameters in the relatively ‘overloaded’ Wk and Wq matrices have larger deviations from a gaussian distribution. I call these matrices ‘overloaded’ for two reasons: (1) because they combine before a nonlinear function (softmax) in a way that resembles a low rank LoRA representation of a larger matrix (exactly identical for identical key and query token indices); and (2) because their outputs need to directly parse the embeddings for both position and token vectorization.

I expect that the highly non-gaussian distribution seen in the first two layers reflects that they are subject to more constraints than later dimensions - the first layers need to interface directly in a lossless way with input that is highly structured and specific on a per-token basis. (unlike later layers that see strongly overlapping outputs from multiple input tokens – see next Section!)  This is also part of a general theme that the first two layers and their outputs look quite different from later layers.


### 5. What do the layer outputs look like?

There’s an infinite amount to say about this, but I want to kick things off by looking at the attention sink phenomenon identified in this paper [<a href = "https://arxiv.org/abs/2309.17453" target = "_blank" rel = "noreferrer noopener">Xiao et al, Sept. 2023</a>]. They observed that attention connecting back to the first token tends to be extremely strong (>~0.5 out of a max of 1) beyond the first two transformer layers. Here’s the figure from their paper:

---
   <img src="/docs/assets/img/Attn-sink-paper-fig2.png" target = "_blank" rel = "noreferrer noopener" alt = "Attention sink paper figure" width="850"/>  

---

This isn’t because of specific information in the first token.  The first token (\<s>) is just a padding token, and the first 1x4096 token output of each layer is the same regardless of the prompt.  Instead, the authors propose: “We attribute the reason to the Softmax operation, which requires attention scores to sum up to one for all contextual tokens. Thus, even when the current query does not have a strong match in many previous tokens, the model still needs to allocate these unneeded attention values somewhere so it sums up to one.”

The figures below show a couple of examples of how this plays out for a specific prompt (see captions):

<img src="/docs/assets/img/Attn-sink-l5.png" target = "_blank" rel = "noreferrer noopener" alt = "Layer 5 attention sink" width="500"/>

**Figure 7: Attention sink effect versus output token.**  Attention to the first token is shown for each attention head in transformer layer #5. The end of each sentence (‘?’ character) is identified with arrows.  The prompt was: "Hey, are you conscious? Can you talk to me?\nI'm not conscious, I think?\nWhat should we talk about?”

We can see the attention sink effect as advertised, but a couple of things jump out:
1. Attention head #15 in layer #5 is represented by a black row.  This means that it’s not sinking attention (much) onto the first token. Instead, the attention from this head goes almost entirely to the token that immediately precedes each query (not shown).
2. The attention sink effect becomes much weaker immediately after the 2nd question mark.  In fact, the effect declines significantly at the end of the first sentence (or 2nd if they’re short) in all of the the tests I’ve done, usually coinciding with a sentence break. Here’s another example. **This suggests that the 1st sentence is viewed differently from others by the model, and may relate to a common role of the first sentence in providing context for everything that follows. Attention to the 1st ‘dummy’ token will cause its value vector (Wv*x1) to be copied over to later token outputs, and may even be something the model uses to highlight meaningful tokens.**

One sees the very similar effects in deeper layers such as layer #25:

<img src="/docs/assets/img/Attn-sink-l25.png" target = "_blank" rel = "noreferrer noopener" alt = "Layer 25 attention sink" width="500"/>

**Figure 8: Attention sink effect versus output token in layer #25.**  Other details are as in Fig. 7.

The first sentence is consistently highlighted as seen in Fig. 7-8, but I haven’t spotted any other highlighted sentences in long prompts.  Instructing the AI to assume a new role (to go from an assistant to a lawyer or famous person, etc), doesn’t seem to do it, and nor does telling it that the next sentence will give it a new role.  Artificially adding a second instance of the first “dummy” token later in the prompt creates a second attention sink, but fails to result in a second highlighted sentence.

OK, so how about the layer output?  The simplest thing to ask is, how similar is the output of a given layer to the output of the next layer at the same token position. Enter Fig. 9:

<img src="/docs/assets/img/self-sim-1.png" target = "_blank" rel = "noreferrer noopener" alt = "1-layer self similarity" width="500"/>

**Figure 9: Self-similarity between the output of adjacent transformer layers.** Normalized inner products between the 4096-long token vectors in each layer and the same token output of the next layer.

Surprisingly, the output is mostly identical from one layer to the next.  If we exclude the first two layers and the last layer, the average normalized inner product between each layer’s input and output is 0.92. The first two layers buck this trend, but they’re highly variable depending on the token (sigma = 0.25 and 0.16). The last layer sees a large drop in correlation to 58% in this example, suggesting some significant final massaging of the vectors before projecting onto the final output token basis.  Still, a naive take would be that the vectors defining word output are mostly settled by several layers before the end of the network.

<img src="/docs/assets/img/self-sim-5.png" target = "_blank" rel = "noreferrer noopener" alt = "1-layer self similarity" width="400"/><img src="/docs/assets/img/self-sim-15.png" target = "_blank" rel = "noreferrer noopener" alt = "1-layer self similarity" width="400"/>

**Figure 10: Self-similarity of token outputs over longer distances.** Equivalent plots to Fig. 9, but with (left) 5- and (right) 15-layer gap between the compared vectors. A dashed prediction curve has been added showing the expected trajectory of the mean curve if single-layer inner products were multiplied together over the indicated distance.

If we skip a few transformers, we find that the numbers decrease as expected for a random cumulative loss of correlation over a few layers, but then hit a plateau (Fig. 7, right – compare with dashed prediction curve). The “attention sink” coupling to the first ‘dummy’ token doesn’t seem to play a direct role in this, as the inner product between the dummy token output and other tokens tends to be just ~10%, and would largely vanish in the inter-layer inner product.

<img src="/docs/assets/img/Same_layer_similarity.png" target = "_blank" rel = "noreferrer noopener" alt = "1-layer self similarity" width="500"/>

**Figure 11: Self-similarity of token outputs in the same layer.** Average of normalized inner products between token outputs of the same layer. “Layer #0” represents the encoded input tokens prior to the first transformer layer. 

Different token outputs of the same layer are also similar (Fig. 11), all the way through the network. Other short prompts (up to ~200 tokens) that I’ve tried yielded essentially identical trends, and even showed some of the same noise-like jitter seen in Fig. 11.

This is getting long, but one last figure! 

<img src="/docs/assets/img/output-amps.png" target = "_blank" rel = "noreferrer noopener" alt = "1-layer self similarity" width="500"/>

***Figure 12: Layer output RMS amplitude and the mean value of RMSNorm scaling parameters.*** 

The amplitude of layer outputs grows throughout most of the network (Fig. 12), and has a striking step anomaly going from the 2nd to 3rd layer.  A first guess would be that growing amplitude will weaken (or break at layer 2-3) the residual connections, but the scaling of transformer input (RMSNorm layer inputs) also grows with depth in the layer, and should counteract some of this effect.  Also, the RMSNorm vector that rescales amplitudes going into the attention block of the first transformer layer are tiny and can even be negative – their interplay with the first two layers warrants a closer look.

### 6. Lessons for LLM architecture

A few impressions:

1. The model is performing analogue computations, so **it is important to understand the effective “noise” that disrupts the fidelity of encoded information.**  There are clear noise issues with the fidelity of information read by the input, output, and middle dictionaries, but one can’t tell if this is noise in the model or just an issue with the dictionary.
   
   A simple interpretation of this would be that because the model is using multiple dictionaries, any incomplete orthogonalization between the dictionary vector spaces will cause words used by one dictionary to come across as noise for all the others.  This gives the model strong motivation to compress the effective rank of each dictionary matrix (such as the 32000x4096 input encoding), though the overlap between the dictionaries also has a meaningful structure as we saw in Fig. 4-5.
   
   In practice the registries are <a href = "/docs/input-vs-output.md" target = "_blank" rel = "noreferrer noopener">not very orthogonal at all (see SVD figure)</a>.  The tradeoff the model seems to have accepted is that each registry can encode words with high specificity, but contains just a few intelligible words and has a high noise floor. (possibly Anoise~0.07 in Fig. 4-5)

2. A brave interpretation of point (1) would be that the **model dimension sets the capacity of the model for internal metacognition**.  If we extrapolate from a noise floor of 0.07, each 4096-long token vector within the model could contain an absolute maximum of ~200 legible embedded words (200 ~ 1/0.07<sup>2</sup>).  However, even the amplitude of ‘illegible’ words could be relevant to performance of the model, as successive transformers adding weakly to the amplitude of an illegible word can eventually bring it through the noise floor.  It’s worth remembering that output amplitudes shift quite slowly from one transformer layer to the next, and the middle of the model (16th layer output) has a ~0.2 average correlation coefficient with the final output.

3. **Attention to the first token – the attention sink phenomenon – seems to act as a highlighter for the first sentence, rather than just a ‘sink’**.  I'm assuming that the first sentence would be highlighted rather than ignored, as it’s the first context the model gets, but it’s also possible that the model has just adapted to ignore redundant initial prompts.

   It would be interesting to try manually manipulating attention to the first token (or weight from the 1st token Wv vector) in the context of prompt engineering, as a way to highlight important instructions.  Looking quickly (<a href = "docs/Llama-2/Attn-sink-vs-output-amp.md" target = "_blank" rel = "noreferrer noopener">plot here</a>), I see that there’s some positive correlation between the attention sink amplitude and output amplitude in the first ~10 layers of the transformer, which could be consistent with a highlighting effect, but later layers have a negative amplitude.

5. The first sentence highlighting function touches on a fascinating question: **how does the model manage so well for vastly different numbers of prompt tokens (say, 20 versus 2000)?**  The model is applying the same kind of processing to inputs regardless of the context length, and
<a href = "https://huggingface.co/blog/tomaarsen/attention-sinks" target = "_blank" rel = "noreferrer noopener">perplexity skyrockets</a> if you try to extend it beyond the trained context window. Amplitude of coupling to the attention sink gradually decays for longer prompts, and it’s easy to speculate at roles that this may be playing to stabilize model behavior.

8. The strong similarity between transformer outputs at different token locations (30+% for later layers [confirmed with a longer ~200 character prompt]) makes me wonder how the model harmonizes its state when information from later tokens sharply contradicts or recontextualizes interpretations based on the earlier tokens.  This sort of scenario seems likely to generate bottlenecks for the masked attention architecture of current generative LLM, in which each token output is unaware of later tokens in the stream.

9. The <a href = "https://arxiv.org/abs/2310.02207" target = "_blank" rel = "noreferrer noopener">first half of the model</a> seems to be specialized in grammatical parsing and memory retrieval, while we mostly see thoughtful updates to the output coming through in the last 1/3rd or so of the transformer layers.

12. The gaussian parameter distributions in the encoder and deeper attention layers is a striking feature, and gaussian distributions are also seen in the feedforward network. My very shallow take is that there’s a vast set of similarly optimized states, and the convergence towards one of these probably looks like a random walk with respect to the basis we’re observing from.

    A corollary to this would be that when the distribution is highly non-gaussian, the solution set is probably more constrained, and one should steer clear of low rank fine tuning techniques like LoRA.  We see this in weight matrices for the 1st two layers, and to some extent for the Wk and Wq matrices throughout the network.


### 7. Useful links

Here are some helpful references:
1. <a href = "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)#Scaled_dot-product_attention" target = "_blank" rel = "noreferrer noopener">Matrix definitions</a> for the attention mechanism.
2. The <a href = "https://arxiv.org/abs/2307.09288" target = "_blank" rel = "noreferrer noopener">Llama 2 release paper</a>.
3. The excellent <a href = "https://www.youtube.com/watch?v=kCc8FmEb1nY" target = "_blank" rel = "noreferrer noopener">“Let’s build GPT” tutorial</a> by Andrej Karpathy (2 hours). 
4. I’d recommend <a href = "https://github.com/ggerganov/llama.cpp" target = "_blank" rel = "noreferrer noopener">starting here</a> if you want to run a quantized Llama-2 model on your own computer.  Here’s a link to <a href = "https://github.com/landrewwray/LLaMA_editor_GUI" target = "_blank" rel = "noreferrer noopener">my bare-bones GUI</a>, which includes AI editor agents for a collaborative text generation experience.
5. Jupyter notebooks used to generate most of the output will be \*linked here\* in a future update.

