from gtda.diagrams import PersistenceEntropy
from gtda.plotting import plot_diagram
from gtda.homology import VietorisRipsPersistence
from generate_datasets import make_point_clouds


n_samples_per_class = 5
point_clouds, labels = make_point_clouds(n_samples_per_class, 10, 0.1)
point_clouds.shape
print(f"There are {point_clouds.shape[0]} point clouds in {point_clouds.shape[2]} dimensions, "
      f"each with {point_clouds.shape[1]} points.")

VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])  # Parameter explained in the text
diagrams = VR.fit_transform(point_clouds)

dp = VR.fit_transform_plot(point_clouds, y=None)

PE = PersistenceEntropy()
features = PE.fit_transform(diagrams)



print("Done")


