## Llama surprises

Image from [PhD Comics](https://phdcomics.com/comics/archive.php?comicid=1959)

<img src="/docs/assets/img/PhD-llama.png" target="_blank" rel="noreferrer noopener" alt="Llama transformer diagram" width="300" />

The last post was a long and technical deep dive into the guts of Llama 2, so I want to zoom out and go over a 
few of the highlights.  The full adventure is [linked here](https://landrewwray.github.io/2023/10/19/Inside-LLaMA-2.html) if you want to take a closer look at the insides of 
this elegant cousin of ChatGPT.

There were some big surprises for me.  For example:

1. The sea of numbers inside the model isn't entirely inscruitable.  It uses several matrices that I'm calling 'dictionaries' to define vocabularies that it uses to store different kinds of information.  These dictionaries are lists that map between meanings (or word tokens) and long state vectors.  The model sends its state vectors through 32 sequential layers of processing (a transformer stack), and if you know the right dictionaries, you can see some of what it's thinking during this process.
    
    I really didn't expect this degree of stability and immediate interpretability across so many neural network layers.  In fact, it turns out that the state of the model changes by very little across a single transformer layer - often by just ~5%.

2. The matrix that's used at the very end of the model to generate output can be used as one dictionary. If we use it 
to generate output from earlier layers of the model, we can 'read its mind' and see how it trying out different 
possibilities for the token to output next. There are at least two other dictionaries that can be used in a similar 
way, but I needed to hack one of them together so it's rather garbled.

3. The model seems to start coming up with possible next tokens from the very beginning, even as it's just starting 
to parse the meaning of what it's working on.  The words of the prompt turn into vectors (via the input dictionary) 
that read as word completions to the output dictionary.  If you turn the word "to" into an input vector and then 
read it with the output dictionary, you get "ast" (as in "toast"), "pping" (as in "topping"), and so on!  The proposed 
words become much more sensical in the last 16 layers.

4. There were some clues as to how the length of the hidden model vectors (which have 4096 numbers each for Llama 2 7B) defines 
the model's potential for internal metacognition -- how many words it can hold in mind at the same time before some 
cross a noise threshold and start to become unintelligible. The answer seems to be something like 100, but the unintelligible
words can sometimes come back into focus if the model reinforces them.  The same kind of noise will also place bounds on the
stability of non-word information, like continuous representations that it seems to have for
<a href = "https://arxiv.org/abs/2310.02207" target = "_blank" rel = "noreferrer noopener">coordinates in time and space</a>.
(Physicist's note: noisy analogue computing systems are much more stable when they're not quantum!!)

5. I saw some neat things about the "attention sink" phenomenon from
<a href = "https://arxiv.org/abs/2309.17453" target = "_blank" rel = "noreferrer noopener">this recent paper</a>.  I'll
add a post soon with some more about this and attention mechanism.
