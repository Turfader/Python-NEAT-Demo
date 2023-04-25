import pygame
import neat
import random
import math
import os


def run_neat(config):
    population = neat.Population(config)

    # Add reporters to show progress in the terminal and save checkpoints
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(10))

    # Run the evolution for up to 100 generations
    winner = population.run(play_game, 100)

    # Show the winning genome's fitness and save it to a file
    print('\nBest genome:\n{!s}'.format(winner))
    with open('best_genome.txt', 'w') as f:
        f.write(str(winner))


class Player:
    def __init__(self):
        self.pos: tuple[int, int] = (50, 50)  # x, y
        self.goals_found: int = 0
        self.dist_to_goal: float = 0.0
        self.final_score: float = 0.0

    def move_up(self):
        if self.pos[1] == 0:
            return
        self.pos = (self.pos[0], self.pos[1]-1)

    def move_down(self):
        if self.pos[1] == 99:
            return
        self.pos = (self.pos[0], self.pos[1]+1)

    def move_left(self):
        if self.pos[0] == 0:
            return
        self.pos = (self.pos[0]-1, self.pos[1])

    def move_right(self):
        if self.pos[1] == 99:
            return
        self.pos = (self.pos[1]+1, self.pos[0])

    def get_dist_to_goal(self, goal) -> float:
        return math.sqrt((self.pos[0] - goal.pos[0])**2 + (self.pos[1] - goal.pos[1])**2)


class Goal:
    def __init__(self):
        self.pos: tuple[int, int] = (random.randint(0, 99), random.randint(0, 99))


def draw_objects(screen, player, goal):
    pygame.draw.circle(screen, (255, 255, 255), player.pos, 5)  # player is a white dot
    pygame.draw.circle(screen, (255, 0, 0), goal.pos, 5)  # goal is a red dot


def play_game(genomes, config):
    pygame.init()
    size = [100, 100]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("NEAT Test")
    game_tick = 0

    for genome_id, genome in genomes:
        player = Player()
        goal = Goal()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        done = False

        while not done:
            clock = pygame.time.Clock()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            inputs = (goal.pos[0]-player.pos[0], goal.pos[1]-player.pos[1], player.get_dist_to_goal(goal))
            output = net.activate(inputs)
            action = output.index(max(output))

            if action == 0:
                player.move_up()
            elif action == 1:
                player.move_down()
            elif action == 2:
                player.move_left()
            elif action == 3:
                player.move_right()

            if player.pos == goal.pos:
                player.goals_found += 1
                goal = Goal()

            player.dist_to_goal = player.get_dist_to_goal(goal)
            player.final_score = player.goals_found + (1 / player.dist_to_goal)

            genome.fitness = player.final_score
            #print(genome.fitness)

            if player.goals_found >= 5:
                genome.fitness += 100
                break


            screen.fill((0, 0, 0))
            draw_objects(screen, player, goal)

            pygame.display.flip()
            #clock.tick(100)

            if game_tick == 1500:
                game_tick = 0
                done = True
                break

            game_tick += 1
            #print("Game tick: ", game_tick)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
