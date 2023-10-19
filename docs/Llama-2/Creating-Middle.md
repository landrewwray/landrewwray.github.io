### “Quick and dirty” recipe for the middle dictionary:

1. Select a list of words for the dictionary.  I chose the “I like the red ball…” prompt and an additional 500 words randomly selected from this list of 3000 common English words.
2. Tokenize the words, and enter each one in a 2-token prompt: [\<s>, I], [\<s>, like], [\<s>, the], etc.
3. Run the model for each 2-token prompt, and save a vector v representing the output of the second layer in the 2nd token location. For each model run:  
    * A. Remove the dummy (\<s>) token component from v, as |v’> = |v> - |dummy><dummy|v>, where <dummy|v> denotes an inner product, and the v and s vectors are L2 normalized beforehand.  Note that <dummy|v’> = 0.  
    * B. Repeat the procedure in step 3A to remove the input dictionary vector component of the 2nd prompt token from v’.  
    * C. Normalize the vector and enter it in the new dictionary!  
