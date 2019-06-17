#Custom Environment
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")


class Blob():
    def __init__(self, SIZE = 10):
        self.size = SIZE
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def act(self, choice, diagonal = False):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if diagonal:

            if choice == 0:
                self.move(x=1, y=1)
            elif choice == 1:
                self.move(x=-1, y=-1)
            elif choice == 2:
                self.move(x=-1, y=1)
            elif choice == 3:
                self.move(x=1, y=-1)

        else:
            print(choice)
            if choice == 0:
                self.move(x=0, y=1)
            elif choice == 1:
                self.move(x=0, y=-1)
            elif choice == 2:
                self.move(x=-1, y=0)
            elif choice == 3:
                self.move(x=1, y=0)


    def move(self, x=-100, y=-100):

        # If no value for x, move randomly
        if x == -100:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if y == -100:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class ENVIRONMENT():



    def __init__(self, player_number=1, enemy_numer=1, food_number=1, size = 10, DIAGONAL = False):
        self.size = size
        self.diagonal = DIAGONAL
        self.player = Blob(size)
        self.enemy = Blob(size)
        self.food = Blob(size)
        self.reward = 0
        self.colors = {1: (255, 0, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.size)
        self.enemy = Blob(self.size)
        self.food = Blob(self.size)
        self.reward = 0

        return (self.player.x, self.player.y), self.reward

    def step(self, action):

        self.player.act(action, self.diagonal)
        self.reward = self.calculate_reward()
        return (self.player.x, self.player.y), self.reward

    def calculate_reward(self):

        if self.player.x == self.enemy.x and self.player.y == self.enemy.y:
            return -100, True

        if self.player.x == self.food.x and self.player.y == self.food.y:
            return 100, True

        else:
            return -1, False


    def render(self):

        env = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.colors[2]
        env[self.player.x][self.player.y] = self.colors[1]
        env[self.enemy.x][self.enemy.y] = self.colors[3]
        img = Image.fromarray(env, 'RGB')
        img = img.resize((300, 300))
        cv2.imshow("image", np.array(img))

    def reset():
        pass
