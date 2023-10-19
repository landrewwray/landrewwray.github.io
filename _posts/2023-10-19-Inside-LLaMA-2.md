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

<img src="/docs/assets/img/Llama-transformer.png" target = "_blank" rel = "noreferrer noopener" alt = "SMBC Sept. 19 2023" width="250"/> &nbsp;&nbsp; <img src="/docs/assets/img/llama-attn-diagram.png" target = "_blank" rel = "noreferrer noopener" alt = "SMBC Sept. 19 2023" width="450"/>
    
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

