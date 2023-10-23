## Llama surprises

Image from [PhD Comics](https://phdcomics.com/comics/archive.php?comicid=1959)

<img src="/docs/assets/img/PhD-llama.png" target="_blank" rel="noreferrer noopener" alt="Llama transformer diagram" width="300" />

The last post was a deep dive into the guts of Llama 2, so I want to zoom out and go over a 
few of the highlights.  The full adventure is [linked here](https://landrewwray.github.io/2023/10/19/Inside-LLaMA-2.html), and I'll revisit some of it in future posts.  Llama 2 is an open source cousin of ChatGPT that follows the same ["decoder-only" development path](https://www.interconnects.ai/p/llm-development-paths).

The high points for me were:

1. It isn't just a sea of numbers inside the model.  There are several [matrices that I'm calling 'dictionaries'](https://landrewwray.github.io/2023/10/19/Inside-LLaMA-2.html#2b-internal-dictionaries-of-an-llm) that map between meanings (or word tokens) and the internal state vectors that are created during text generation.  The model can remember several words from each dictionary by adding these state vectors together.  For example, if a state vector of the model is created by adding four word vectors together (say, **v = v_red + v_apple + v_purple + v_plum**), the model will effectively store an unordered list of those four words ({'red', 'apple', 'purple', 'v_plum'}). If two of the words come from one dictionary (say, **v_red** and **v_apple** in dictionary \#1) while the other two are from another dictionary (**v_purple**, **v_plum** in dictionary \#2), the model will be able to work separately with those word pairs.
  
   The model sends its state vectors through 32 sequential layers of processing (a transformer stack), and if you know the right dictionaries, you can see some of what it's thinking during this process.  I really didn't expect this degree of stability and interpretability across so many neural network layers.  In fact, it turns out that the state of the model changes by very little across a single transformer layer - often by just ~5%.

3. The matrix that's used at the very end of the model to generate output is the most interesting dictionary, and one of the most accessible for research. If we use this to interpret state vectors within earlier layers of the model, we can 'read its mind' and see how it tries out different 
possibilities for the next output token. For example, if the prompt says you "go to the basketball court to" - the model will begin by coming up with well motivated but incorrect extensions of the text like "oast", as in "toast". After 8 layers we see "to hold", after 12: "to test"; after 16: "to exercise"; after 20: "to ball" and "to play"... until by layer 32, it has a long list of sensible next words, starting with "play", "shoot", "practice", "have", and "throw". There are at least two other dictionaries that can be used in a similar way, one being the input encoding matrix that holds a memory of the original prompt. The third dictionary stores closely associated input words, but can't be very well understood yet, as all I have for it at the moment is a hacked together first draft.

4. The model actually starts coming up with possible next tokens *even before the first transformer layer!*  The words of the prompt turn into vectors (via the input dictionary) that read as word completions to the output dictionary.  If you turn the word "to" into an input vector and then 
read it with the output dictionary, you get "ast" (as in "toast"), "pping" (as in "topping"), and so on!  The input encoding vector of "play" reads as words like "ground", "offs", and "test" in the output dictionary.

5. There were some clues as to how the length of the state vectors helps define a key aspect of [the model's 'cognitive capacity'](https://landrewwray.github.io/2023/10/19/Inside-LLaMA-2.html#6-lessons-for-llm-architecture): how many words it can hold in mind at the same time and think about relationally before some cross a noise threshold and start to become unintelligible. The answer seems to be something like 100 for Llama 2 7B (with 4096 numbers per vector), but even unintelligible words can come back into focus if the model reinforces them.  The same kind of noise will also place bounds on the
stability of non-word information, like continuous representations that it seems to have for
<a href = "https://arxiv.org/abs/2310.02207" target = "_blank" rel = "noreferrer noopener">coordinates in time and space</a>.
(Physicist's note: noisy analog systems can be much more stable when they're not quantum computers!!)

6. A lot of things were only touched on, and will be interesting to look at more in the future.  I saw some neat things about the "attention sink" phenomenon from
<a href = "https://arxiv.org/abs/2309.17453" target = "_blank" rel = "noreferrer noopener">this recent paper</a>.  I'll
add a post soon with some more about this and the attention mechanism.
