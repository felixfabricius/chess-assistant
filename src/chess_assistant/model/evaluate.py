def evaluate(model, dataloader, loss_fn):
    ### Calculate:
        # losses for each individual datapoint;; perhaps averaged across batch
        # accuracy for each individual square
    
    ### On position level
        # check if current position is valid - this should also be returned from dataloader
        # if it is; then return previous fen
        # then get the machinery going: predict all the square in one board position
        # note: this dataloader should therefore perhaps operate at this slightly higher level?
        # i.e. we always want to make predicitons for al the squares of a board - don't want to mix 
        # between boards.
    
    
        # 

