# stringart_app/image_to_vector_algorithms/memetic.py

import random
import logging
from typing import List, Dict, Optional, Callable
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
    ELITE_FRACTION = 0.3  # fraction of population preserved without change

    def generate(
        self,
        pixels: np.ndarray,
        n_anchors: int = 180,
        n_strings: int = 200,
        line_thickness: int = 1,
        sample_pairs: int = 1000,
        logger: Optional[logging.Logger] = None,
        *,
        vector_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, int]]:
        if logger is None:
            logger = logging.getLogger(__name__)

        height, width = pixels.shape
        logger.debug(f"[memetic] Starting with anchors={n_anchors}, strings={n_strings}")

        # Precompute anchors and all possible pairs
        anchors = generate_radial_anchors(n_anchors, width, height, logger=logger)
        all_pairs: List[tuple[int, int]] = [
            (i, j) for i in range(n_anchors) for j in range(i + 1, n_anchors)
        ]
        genome_length = len(all_pairs)
        logger.debug(f"[memetic] Total candidate pairs: {genome_length}")

        # Precompute binary masks for all pairs
        masks_flat = []
        for (i, j) in all_pairs:
            img = Image.new('L', (width, height), color=0)
            draw = ImageDraw.Draw(img)
            draw.line([anchors[i], anchors[j]], fill=255, width=line_thickness)
            masks_flat.append(np.array(img, dtype=bool).ravel())
        logger.debug(f"[memetic] Precomputed {len(masks_flat)} masks")

        # Convert target to float array for SSE computation
        target_flat = pixels.astype(np.float32).ravel()

        def render_score(chrom: List[int]) -> float:
            """
            Draws each selected line onto a blank canvas and computes
            squared-error against the target image. Returns SSE as Python float.
            """
            canvas_flat = np.full_like(target_flat, 255.0, dtype=np.float32)
            for gene in chrom:
                canvas_flat[masks_flat[gene]] = 0.0  # draw black lines
            residual = canvas_flat - target_flat
            return float(np.sum(residual ** 2))

        # Initialize population: each individual is a random set of unique genes
        population: List[List[int]] = [
            random.sample(range(genome_length), n_strings)
            for _ in range(self.POP_SIZE)
        ]
        logger.debug(f"[memetic] Initialized population of size {self.POP_SIZE}")

        # Evolutionary loop
        for gen in range(self.GENERATIONS):
            population.sort(key=render_score)
            best_sse = render_score(population[0])
            logger.debug(f"[memetic] Generation {gen+1}: best SSE={best_sse:.2f}")

            elite_size = max(1, int(self.POP_SIZE * self.ELITE_FRACTION))
            next_gen = population[:elite_size]

            while len(next_gen) < self.POP_SIZE:
                parent1, parent2 = random.sample(population[:10], 2)
                crossover_point = random.randint(1, n_strings - 1)
                child = parent1[:crossover_point] + [
                    gene for gene in parent2 if gene not in parent1[:crossover_point]
                ]
                for idx in range(n_strings):
                    if random.random() < self.MUTATION_RATE:
                        child[idx] = random.randrange(genome_length)
                seen: set[int] = set()
                for idx in range(len(child)):
                    if child[idx] in seen:
                        replacement = random.randrange(genome_length)
                        while replacement in seen:
                            replacement = random.randrange(genome_length)
                        child[idx] = replacement
                    seen.add(child[idx])
                next_gen.append(child)

            population = next_gen

        best_genome = population[0]
        logger.debug(f"[memetic] Finished; best SSE={render_score(best_genome):.2f}")

        # Convert gene indices back to vectors, streaming via callback if provided
        vectors: List[Dict[str, int]] = []
        for gene in best_genome:
            i, j = all_pairs[gene]
            vectors.append({"from": i, "to": j})
            if vector_callback:
                vector_callback(i, j)

        return vectors
