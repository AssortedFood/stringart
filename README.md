# StringArt

StringArt is a Django-based web app that converts grayscale images into “string art” renderings using a variety of algorithms (greedy, coverage, graph-optimisation, memetic, simulated annealing, etc.), with live physics-based previews in the browser.

## 📖 Overview

Users upload one or more images, choose which string-art algorithms to run, and watch each stage stream into the page:

1. **Grayscale pass**: quantize the image to N gray levels.  
2. **Algorithm phases**: for each selected algorithm, generate nails-to-nails vectors and overlay a live physics simulation.  
3. **Logs pane**: real-time console output of the generation process via SSE.

## 🚀 Features

- Multiple string-art strategies: **greedy**, **coverage**, **graph-optimisation**, **hough-greedy**, **memetic**, **simulated-annealing**  
- Live physics preview (Verlet springs) in the browser  
- Server-Sent Events for real-time logs and results  
- Dockerized with a single-process, multi-threaded Gunicorn + WhiteNoise setup  

## 🚧 Installation

```bash
git clone git@github.com:AssortedFood/stringart.git
cd stringart
````

1. **Install Python deps**

   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```
2. **Collect static assets**

   ```bash
   python manage.py collectstatic --no-input
   ```
3. **Run tests**

   ```bash
   pytest
   ```

## 📂 Project Structure

```text
.
├── docker-compose.yml
├── deploy.sh
├── Dockerfile
├── .dockerignore
├── manage.py
├── requirements.txt
├── stringart_app/
│   ├── image_to_vector_algorithms/
│   │   ├── base.py
│   │   ├── coverage.py
│   │   ├── greedy.py
│   │   └── …  
│   ├── preprocessing.py
│   ├── planner.py
│   ├── renderer.py
│   ├── views.py
│   └── tests/
├── stringart_project/
│   ├── settings.py
│   ├── wsgi.py
│   └── asgi.py
└── templates/
    └── core/home.html
```

* **`image_to_vector_algorithms/`**: strategy implementations
* **`preprocessing.py`**: load → grayscale → quantize
* **`planner.py`**: dispatch to chosen algorithm
* **`renderer.py`**: static preview & overlay functions
* **`views.py`**: upload, SSE log/result streaming, orchestrates phases
* **`tests/`**: unit tests for each core module


## 📄 License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.