# TODO
- More data
    - Separate testing chunks as random chunks instead of last chunk
- Calculate theoretical maximum profit
- Save Networks

# Fixes to previous bugs
- tanh vs sigmoid speed/efficacy

# TO think about?
- When to use different chunks of data?
- Drop out
- MAX_CYCLES_ADD_EDGE should be V^2 instead of V

# Done
- Fix disabled/enabled edges in crossing over
- mutate_add_edge: Prevent adding self loops, prevent adding multiple edges, optimize by doing dictionaries
- initial population shoudl be divided into species
- Limit species size ?????
- Limit genome size !!!
    - Node mutation number??
- Fix fitness sharing
- Allow Network to Prune itself
- Rewrite population code
- Test time_step environment
    - input 5 t_1 output t_2 5