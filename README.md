# MathQuest

Streamlit practice app for first-year engineering mathematics and EG1011 Statics and Dynamics.

## Topics
- Linear equations
- Trigonometry
- Vector components
- Differentiation
- Integration
- Centroid and second moment of area using integration

## Centroid module
Six scaffolded levels use programmatically generated Matplotlib diagrams. Students move vertical or horizontal differential strips, view strip dimensions and centroids, choose the integration method, and solve area, centroid, reference-axis second moment, and centroidal second-moment questions.

## Deploy from GitHub
1. Upload the contents of this package to a GitHub repository.
2. In Streamlit Community Cloud, select the repository.
3. Set the main file to `app.py`.
4. Deploy.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Optional JCU logo
Place an approved logo at `assets/jcu_logo.png`. The app displays a text fallback if the image is absent.
