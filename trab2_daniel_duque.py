import pandas as pd
from scipy.stats import ttest_rel as t_test
from scipy.stats import wilcoxon
from scipy.stats import truncnorm as trunc_nomal
import math
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pygame
import os
import random
import time
from sys import exit
import seaborn as sns

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join(
                    "Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join(
                    "Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join(
                    "Assets/Cactus", "LargeCactus3.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

    def getXY(self):
        return (self.dino_rect.x, self.dino_rect.y)


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle():
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < - self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return (self.type)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 19:
            self.index = 0
        SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1


def first(x):
    return x[0]


def fitness(x):
    return x.getFitness()

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


# OTHER CONSTANTS
MIN = 60
HOUR = 60*60
DAY = 24*60*60

N_PLAYS = 1

# GA CONSTANTS
RAND_RANGE = 100
MUT_RANGE = 50

IND_SIZE = 39
POP_SIZE = 100

ELITE_RATIO = 0.20
CROSS_RATIO = 0.90
MUT_RATIO = 0.50

MAX_ITER = 10_000
MAX_TIME = 1*HOUR


class Individual():

    def __init__(self, weights=None, fitness=None, ind_size=IND_SIZE):
        if weights is None:
            self.weights = [random.randint(-RAND_RANGE, RAND_RANGE)
                            for i in range(ind_size)]
        else:
            self.weights = weights

        if fitness is None:
            global aiPlayer
            aiPlayer = NeuralNetwork(self.weights)
            _, value = manyPlaysResults(N_PLAYS)
            self.fitness = int(value)
        else:
            self.fitness = fitness

        # print(f'f: {self.fitness} w: {self.weights}')

    def __str__(self):
        return f'fitness = {self.fitness} - weights = {self.weights}'

    def getFitness(self):
        return self.fitness

    def getWeights(self):
        return self.weights.copy()

    def mutateIndiv(self, mut_ratio=MUT_RATIO):
        # print('Mutate Individual')
        for i, v in enumerate(self.weights):
            rand = random.uniform(0, 1)
            if rand <= mut_ratio:
                offset = random.randint(-MUT_RANGE, MUT_RANGE)
                self.weights[i] = v + offset

    def crossoverIndiv(self, indiv):
        # print('Crossover Individual')
        w1 = self.getWeights()
        w2 = indiv.getWeights()

        r = random.randint(0, len(w1) - 1)
        w1_crossed = w1[:r] + w2[r:]
        w2_crossed = w2[:r] + w1[r:]

        self.weights = w1_crossed
        indiv.weights = w2_crossed

    def calculateFitnessIndiv(self):
        global aiPlayer
        aiPlayer = NeuralNetwork(self.weights)
        _, value = manyPlaysResults(N_PLAYS)
        self.fitness = int(value)


def roulette_construction(population):
    aux_pop = []
    roulette = []
    pop = population.pop
    total_fitness = population.sumPopFitness()

    for indiv in pop:
        if total_fitness != 0:
            ratio = indiv.getFitness() / total_fitness
        else:
            ratio = 1
        aux_pop.append((ratio, indiv))

    acc_value = 0
    for ratio, indiv in aux_pop:
        acc_value = acc_value + ratio
        s = (acc_value, indiv)
        roulette.append(s)
    return roulette


def roulette_run(rounds, roulette):
    if roulette == []:
        return []
    selected = []
    while len(selected) < rounds:
        r = random.uniform(0, 1)
        for ratio, indiv in roulette:
            if r <= ratio:
                f = indiv.getFitness()
                w = indiv.getWeights()
                selected.append(Individual(weights=w, fitness=f))
                break

    return selected


good_indiv_123 = [-84, 31, -54, -58, 45, -42, -93, 65, -78, -82, -46, 29, 34, -84, 73, 64, 83,
                  45, -46, 47, -68, -52, 79, -73, 65, -71, -19, 69, 105, -61, -110, 7, 11, -89, -13, -71, -82, 61, 94]

good_indiv_200 = [34, -2, 4, 73, -93, -93, -16, -59, -4, 83, -38, -31, 82, 59, 84, -15, -100,
                  10, -59, 56, 105, 25, -35, -11, -47, 15, -82, -85, -71, -21, -52, -14, -76, -80, -51, 24, -74, -64, -45]

good_indiv_1295 = [34, -2, 4, 73, -93, -93, -25, -50, -4, 83, -38, -33, 82, 68, 89, -15, -100, 10,
                   -65, 62, 111, 23, -35, -13, -57, 16, -85, -89, -72, -21, -46, -14, -71, -80, -51, 18, -74, -55, -47]

good_indiv_1133 = [34, -2, 4, 73, -93, -93, -16, -59, -4, 83, -38, -31, 82, 59, 84, -15, -100, 10,
                   -59, 56, 105, 25, -35, -11, -47, 15, -82, -85, -71, -21, -52, -14, -76, -80, -51, 24, -74, -64, -45]

good_indiv_1111 = [34, -2, 4, 73, -97, -93, -22, -59, -4, 88, -48, -31, 82, 65, 88, -15, -104, 9,
                   -59, 61, 105, 25, -35, -15, -47, 15, -75, -85, -80, -23, -42, -14, -76, -80, -51, 24, -83, -64, -41]

good_indiv_1025 = [35, -2, 4, 78, -83, -87, -16, -59, -4, 91, -29, -37, 82, 62, 86, -15, -100, 10,
                   -59, 56, 105, 17, -35, -11, -42, 15, -78, -94, -71, -21, -53, -14, -76, -80, -43, 16, -76, -64, -45]

good_indiv_1765 = [41, -2, 1, 41, -223, -59, -34, -165, -78, 48, -29, -168, 59, -4, 137, -8, -7, -39,
                   -110, 104, 61, 21, -18, -53, -9, -14, -100, -127, -138, 84, -84, -40, -72, -75, -131, -34, -1, -26, 19]


class Population(Individual):

    def __init__(self, pop=None, pop_size=POP_SIZE):
        super().__init__()
        if pop is None:
            # self.pop = [Individual() for _ in range(pop_size)]
            self.pop = [Individual(good_indiv_1765) for _ in range(pop_size)]
        else:
            self.pop = pop

        self.pop_size = len(self.pop)

    def __str__(self):
        print(f'POPULATION:\n')
        for index, indiv in enumerate(self.pop):
            print(f'Individual {index}: {indiv}')
        return ''

    def getPopSize(self):
        return self.pop_size

    def getPop(self):
        return self.pop

    def serializePop(self, gen):
        file = open('generations.txt', 'a')
        mean_f = self.meanPopFitness()
        best_f = self.bestPopFitness().getFitness()
        file.write(
            f'Gen: {gen} - Mean Fitness: {mean_f} - Best Fitness: {best_f}\n')

        pop = sorted(self.pop, key=fitness, reverse=True)
        for individual in pop:
            w = individual.getWeights()
            f = individual.getFitness()
            file.write(f'f: {f} - w: {w}\n')
        file.write(f'\n')
        file.close()

    def sumPopFitness(self):
        return sum(individual.getFitness() for individual in self.pop)

    def meanPopFitness(self):
        return int(np.mean([individual.getFitness() for individual in self.pop]))

    def bestPopFitness(self):
        return sorted(self.pop, key=fitness, reverse=True)[0]

    def calculateFitnessPop(self):
        for individual in self.pop:
            individual.calculateFitnessIndiv()

    def isConvergentPop(self):
        if self.pop != []:
            base = self.pop[0]
            i = 0
            while i < self.pop_size:
                if base != self.pop[i]:
                    return False
                i += 1
            return True

    def mergeTwoPop(self, other):
        p1 = self.pop
        p2 = other.pop
        new_pop = p1 + p2
        new_size = len(new_pop)
        return Population(pop=new_pop, pop_size=new_size)

    def elitePop(self, elite_ratio=ELITE_RATIO):
        # print('Elite Population\n')
        n = math.floor((elite_ratio/100)) * self.pop_size
        n = 1 if n < 1 else n

        aux_pop = []
        for individual in self.pop:
            w = individual.getWeights()
            f = individual.getFitness()
            aux_pop.append(Individual(weights=w, fitness=f))

        elite_pop = sorted(aux_pop, key=fitness, reverse=True)[:n]
        return Population(pop=elite_pop, pop_size=len(elite_pop))

    def selectPop(self, n):
        # print('Select Population\n')
        aux_pop = roulette_construction(self)
        new_pop = roulette_run(n, aux_pop)
        return Population(pop=new_pop, pop_size=len(new_pop))

    def mutatePop(self, mut_ratio=MUT_RATIO):
        # print('Mutate Population\n')
        for individual in self.pop:
            individual.mutateIndiv(mut_ratio)

    def crossoverPop(self, cross_ratio=CROSS_RATIO):
        # print('Crossover Population\n')
        for _ in range(round(self.pop_size/2)):
            rand = random.uniform(0, 1)
            i0 = random.randint(0, self.pop_size - 1)
            i1 = random.randint(0, self.pop_size - 1)
            parent1 = self.pop[i0]
            parent2 = self.pop[i1]

            if rand <= cross_ratio:
                parent1.crossoverIndiv(parent2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork():
    def __init__(self, weights):
        self.weights = weights

    def layerOperation(self, layer, weights, layer_size):
        new_layer = []
        start = 0
        end = 0
        for i in range(layer_size):
            start = i * layer_size
            end = (i+1) * layer_size

            node = np.dot(layer, weights[start:end])
            node = sigmoid(node)
            new_layer.append(node)

        return np.array(new_layer)

    def keySelector(self, distance, obHeight, speed, obType):
        input_vector = [speed, distance, obHeight]
        weights = self.weights

        # print(weights, len(weights))
        hidden_1 = self.layerOperation(input_vector, weights[:9], 3)  # 3*3
        hidden_2 = self.layerOperation(hidden_1, weights[9:18], 3)  # 3*3
        hidden_3 = self.layerOperation(hidden_2, weights[18:27], 3)  # 3*3
        hidden_4 = self.layerOperation(hidden_3, weights[27:36], 3)  # 3*3
        output_node = sigmoid(np.dot(hidden_4, weights[36:39]))  # 3
        # print(output_node)

        if output_node < 0.5:
            return 'K_UP'
        else:
            return 'K_DOWN'

    def updateState(self, weights):
        self.weights = weights


class KeySimplestClassifier(NeuralNetwork):
    def __init__(self, state):
        self.state = state
        print(self.state)

    def keySelector(self, distance, obHeight, speed, obType):
        s, d = self.state

        if speed < s:
            limDist = d
        else:
            return 'K_NO'

        if distance <= limDist:
            if isinstance(obType, Bird) and obHeight > 50:
                return "K_DOWN"
            else:
                return "K_UP"

    def updateState(self, state):
        self.state = state


def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    cloud = Cloud()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    death_count = 0
    spawn_dist = 0

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        SCREEN.fill((255, 255, 255))

        distance = 1500
        obHeight = 0
        obType = 2
        if len(obstacles) != 0:
            xy = obstacles[0].getXY()
            distance = xy[0]
            obHeight = obstacles[0].getHeight()
            obType = obstacles[0]

        if GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()
        else:
            userInput = aiPlayer.keySelector(
                distance, obHeight, game_speed, obType)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        player.update(userInput)
        player.draw(SCREEN)

        for obstacle in list(obstacles):
            obstacle.update()
            obstacle.draw(SCREEN)

        background()

        cloud.draw(SCREEN)
        cloud.update()

        score()

        # clock.tick(60)
        pygame.display.update()

        for obstacle in obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                # pygame.time.delay(2000)
                death_count += 1
                return points


def manyPlaysResults(rounds):
    # print(f'Playing {rounds} rounds...\n')
    results = []
    for round in range(rounds):
        results += [playGame()]
    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())


def init_ga(pop_size=POP_SIZE):
    start = time.process_time()

    pop = Population(pop_size=pop_size)
    opt_indiv = pop.bestPopFitness()
    iter = 0
    iter_v = []
    fitness_v = []

    end = start
    return start, end, iter, pop, opt_indiv, fitness_v, iter_v


def genetic_algorithm(pop_size=POP_SIZE, cross_ratio=CROSS_RATIO, mut_ratio=MUT_RATIO, elite_ratio=ELITE_RATIO, max_iter=MAX_ITER, max_time=MAX_TIME):
    print('\n\t--------- GA ---------\n')

    start, end, iter, pop, opt_indiv, fitness_v, iter_v = init_ga(pop_size)

    while iter < max_iter and end-start <= max_time:
        print(f'\n\tGA: iter: {iter} | time: {int(end-start)} s')

        pop.serializePop(iter)
        iter_v.append(iter)
        fitness_v.append(opt_indiv.getFitness())

        best_indiv = pop.bestPopFitness()
        best_w = best_indiv.getWeights()
        best_f = best_indiv.getFitness()

        print(f'\n\tmean fit: {pop.meanPopFitness()} - best fit: {best_f}')

        elite_pop = pop.elitePop(elite_ratio=elite_ratio)

        if best_f > opt_indiv.getFitness():
            opt_indiv = Individual(weights=best_w, fitness=best_f)

        selected_pop = pop.selectPop(pop_size - elite_pop.getPopSize())
        selected_pop.crossoverPop(cross_ratio=cross_ratio)
        selected_pop.mutatePop(mut_ratio=mut_ratio)
        selected_pop.calculateFitnessPop()
        pop = elite_pop.mergeTwoPop(selected_pop)

        iter += 1
        end = time.process_time()

    # print(f'OPT INDIV:', end=' ')
    # print(opt_indiv)

    plot_evolution(iterations=iter_v, fitness=fitness_v)

    return opt_indiv, iter, iter_v, fitness_v


def plot_evolution(iterations, fitness):
    sns.lineplot(x=iterations, y=fitness)
    plt.xlabel('iterations')
    plt.ylabel('score')
    plt.savefig('fitness_evolution.pdf')


confiance_interval = 0.95  # 95%
alpha = 1 - confiance_interval  # 5%


def results_paired(scores1, scores2):
    _, p_value_t = t_test(scores1, scores2)
    _, p_value_w = wilcoxon(scores1, scores2)

    return p_value_t, p_value_w


def table_paired(scores):
    # lista as keys para facilitar manipulacao
    keys = list(scores.keys())
    # quantidade de keys para criar tamanho da matriz
    size = len(keys)
    # matrix que eh a tabela pareada
    matrix_hyp = np.zeros((size, size))

    # faz a tabela pareada
    for i in range(size):
        k1 = keys[i]
        s1 = scores[k1]
        for j in range(size):
            k2 = keys[j]
            s2 = scores[k2]
            # diagonal vai ficar o nome do classificador
            if k1 == k2:
                matrix_hyp[i][j] = np.NaN
            # nao diagonal vai ficar os p-values dos testes pareados
            else:
                t, w = results_paired(s1, s2)
                # triangular superior = p-value do t-pareado
                matrix_hyp[i][j] = t
                # triangular inferior = p-value do wilcoxon
                matrix_hyp[j][i] = w

        # transforma a matriz em um dataframe para facilitar manipulacao
        df = pd.DataFrame(matrix_hyp, index=scores, columns=scores)
        # arredonda os valores para 3 casas decimais
        df = df.applymap(lambda x: f'{x:.3e}' if x < 0.0001 else f'{x:.3f}')
        # transforma os valores para string para poder manipular
        df = df.astype(str)
        # coloca a diagonal do dataframe com os respectivos nomes dos classificadores
        df.at['DUQUE', 'DUQUE'] = 'DUQUE'
        df.at['VAREJAO', 'VAREJAO'] = 'VAREJAO'

        # coloca em negrito quando p-value < 0.05
        def apply_bold(txt):
            try:
                bold = 'bold' if float(txt) < alpha else ''
                return 'font-weight: %s' % bold
            except:
                None

        # aplica o negrito no estilo do dataframe
        df_style = df.style.applymap(apply_bold)

        return df, df_style


def main():
    global aiPlayer

    file = open('generations.txt', 'w')
    file.write('')
    file.close()

    best_individual, iter, iter_v, fitness_v = genetic_algorithm()
    aiPlayer = NeuralNetwork(best_individual.getWeights())

    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std())
    print(value)

    file = open('generations.txt', 'a')
    file.write(
        f'\nFINAL STATES: Iter: {iter}\nopt_indiv: {best_individual}\n')
    file.write(
        f'\nFINAL RESULTS:\n {res}\nMean: {npRes.mean()}\nStd: {npRes.std()}\nValue: {value}')

    # abaixo esta a parte só pra gerar a tabela pareada e o boxplot usados no artigo

    duque_scores = [1547.25, 51.0, 302.5, 446.75, 607.75, 58.75, 50.25, 159.75, 1819.5, 1661.75, 23.25, 100.25, 693.0, 1823.25,
                    571.5, 100.5, 499.25, 1528.0, 588.75, 245.5, 503.75, 60.75, 37.25, 539.5, 286.5, 1007.5, 258.5, 236.5, 425.25, 83.5]

    varejao_scores = [1214.0, 759.5, 1164.25, 977.25, 1201.0, 930.0, 1427.75, 799.5, 1006.25, 783.5, 728.5, 419.25, 1389.5, 730.0, 1306.25,
                      675.5, 1359.5, 1000.25, 1284.5, 1350.0, 751.0, 1418.75, 1276.5, 1645.75, 860.0, 745.5, 1426.25, 783.5, 1149.75, 1482.25]

    dict_scores = {'DUQUE': np.array(duque_scores),
                   'VAREJAO': np.array(varejao_scores)}

    df = pd.DataFrame(dict_scores)

    fig, axes = plt.subplots(figsize=(7, 7))
    sns.boxplot(data=df.loc[:, :])
    axes.set_xlabel('Classificador')
    axes.set_ylabel('Acurracy Score')
    plt.savefig('accuracy_boxplot.pdf')

    df_paired, df_style_paired = table_paired(dict_scores)
    with pd.ExcelWriter('paired_table.xlsx') as writer:
        df_paired.to_excel(writer, sheet_name='sheet_1')


main()
