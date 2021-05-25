'''
A standard deck of playing cards has four suits (Hearts, Diamonds, Spades and Clubs)
and thirteen ranks (2 through 10, then the face cards Jack, Queen, King and Ace)
for a total of 52 cards per deck. Jacks, Queens and Kings all have a rank of 10.
Aces have a rank of either 11 or 1 as needed to reach 21 without busting.
As a starting point in your program, you may want to assign variables
to store a list of suits, ranks, and then use a dictionary to map ranks to values.

here we are using tuple to store the cards and ranks since they are immutable
'''

'''
mports and Global Variables
Step 1: Import the random module. This will be used to shuffle the deck prior to dealing. 
Then, declare variables to store suits, ranks and values. 
You can develop your own system, or copy ours below. 
Finally, declare a Boolean value to be used to control while loops. 
This is a common practice used to control the flow of the game.
'''
import random
suits = ('Hearts', 'Diamonds', 'Spades', 'Clubs')
ranks = ('Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace')
values = {'Two':2, 'Three':3, 'Four':4, 'Five':5, 'Six':6, 'Seven':7, 'Eight':8, 'Nine':9, 'Ten':10, 'Jack':10,
         'Queen':10, 'King':10, 'Ace':11}
playing=True

'''
Step 2: Create a Card Class
A Card object really only needs two attributes: suit and rank. 
You might add an attribute for "value" - we chose to handle value later when developing our Hand class.
In addition to the Card's __init__ method, consider adding a __str__ method that, 
when asked to print a Card, returns a string in the form "Two of Hearts"
'''

class Card:

    def __init__(self,suit,rank):
        self.suit=suit
        self.rank=rank

    def __str__(self):
        return self.suit +'of'+ self.rank

class Deck:

    def __init__(self):
        self.deck=[] # to store all 52 card objects in list
        for suit in suits:
            for rank in ranks:
                self.deck.append(Card(suit,rank))   # build Card objects and add them to the list

    def __str__(self):
        deck_comp=''   # to show the card composition first empty string created and card str show method called
        for card in self.deck:
            deck_comp+='\n'+card.__str__()     # add each Card object's print string
        return 'The deck has : '+deck_comp

    def shuffle(self):
        random.shuffle(self.deck)

    def deal(self):
        single_card=self.deck.pop()
        return single_card
test_deck=Deck()
test_deck.shuffle()
print(test_deck)
