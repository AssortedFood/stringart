# stringart_app/image_to_vector_algorithms/memetic.py

import random
from typing import List, Dict
import numpy as np
from PIL import Image, ImageDraw

from .base import StringArtAlgorithm
from ..renderer import generate_radial_anchors


class MemeticAlgorithm(StringArtAlgorithm):
    """
    Genetic algorithm with local greedy “repair” on each offspring.
    Chromosomes are lists of anchor-pair indices.
    """
    POP_SIZE = 30
    GENERATIONS = 100
    MUTATION_RATE = 0.1

    def generate(
        self,
        pixels: np.ndarray,
        n_anchors: int = 180,
        n_strings: int = 200,
        line_thickness: int = 1,
        sample_pairs: int = 1000
    ) -> List[Dict[str, int]]:
        # Dimensions
        height, width = pixels.shape

        # Precompute anchors and all possible pairs
        anchors = generate_radial_anchors(n_anchors, width, height)
        all_pairs: List[tuple[int, int]] = [
            (i, j) for i in range(n_anchors) for j in range(i + 1, n_anchors)
        ]

        # Convert input to int16 for rendering math
        target = pixels.astype(np.int16)

        def render_score(chrom: List[int]) -> int:
            """
            Draws each selected line onto a blank canvas and computes
            squared-error against the target image. Returns a Python int.
            """
            canvas = np.full_like(pixels, 255, dtype=np.int16)

            for gene in chrom:
                i, j = all_pairs[gene]
                # draw onto a PIL image for antialiasing consistency
                pil = Image.fromarray(canvas.astype(np.uint8), mode='L')
                draw = ImageDraw.Draw(pil)
                draw.line([anchors[i], anchors[j]], fill=0, width=line_thickness)
                # bring it back into our numpy canvas
                canvas = np.array(pil, dtype=np.int16)

            # compute SSE and convert to int
            sse = np.sum((canvas - target) ** 2)
            return int(sse)

        # Initialize population: each individual is a random set of unique genes
        genome_length = len(all_pairs)
        population: List[List[int]] = [
            random.sample(range(genome_length), n_strings)
            for _ in range(self.POP_SIZE)
        ]

        # Evolutionary loop
        for _ in range(self.GENERATIONS):
            # Evaluate & sort by ascending error (lower is better)
            population = sorted(population, key=render_score)

            # Elitism: keep top half
            next_gen = population[: self.POP_SIZE // 2]

            # Fill back up with offspring
            while len(next_gen) < self.POP_SIZE:
                parent1, parent2 = random.sample(population[:10], 2)
                # One-point crossover
                crossover_point = random.randint(1, n_strings - 1)
                child = parent1[:crossover_point] + [
                    gene for gene in parent2 if gene not in parent1[:crossover_point]
                ]

                # Mutation: random swaps
                for idx in range(n_strings):
                    if random.random() < self.MUTATION_RATE:
                        child[idx] = random.randrange(genome_length)

                # Local repair: fix duplicates
                seen: set[int] = set()
                for idx in range(len(child)):
                    if child[idx] in seen:
                        # pick a replacement not seen yet
                        candidates = random.sample(range(genome_length), 20)
                        for c in candidates:
                            if c not in seen:
                                child[idx] = c
                                break
                    seen.add(child[idx])

                next_gen.append(child)

            population = next_gen

        # Best individual is first in the final sorted population
        best_genome = population[0]

        # Convert gene indices back to {"from": i, "to": j} dicts
        return [
            {"from": all_pairs[g][0], "to": all_pairs[g][1]}
            for g in best_genome
        ]
