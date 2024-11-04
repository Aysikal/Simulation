import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
import time

start_time = time.time()
G = 6.67428e-11
size = 500
furthest_object = 100
no_astroids = 100
frames = 1000
AU = 1.496 * 10**11
collision_distance = 1 * AU  # Example collision distance
MIN_VELOCITY = 1  # Minimum velocity in m/s

class Earth_and_sun:
    AU = 1.496 * 10**11
    def __init__(self, x, y, mass):
        self.x = x
        self.y = y
        self.mass = mass
        self.distance_to_sun = 0
        self.x_vel = 0
        self.y_vel = 0
        self.size = 2  # Default size for the objects
    def attraction(self, other):
        distance_x = other.x - self.x
        distance_y = other.y - self.y
        distance = np.sqrt(distance_x**2 + distance_y**2)
        force = G * self.mass * other.mass / distance**2
        theta = np.arctan2(distance_y, distance_x)
        force_x = np.cos(theta) * force
        force_y = np.sin(theta) * force
        return force_x, force_y
    def update_position(self, objects, delta_t):
        total_fx = total_fy = 0
        for planet in objects:
            if self == planet:
                continue
            fx, fy = self.attraction(planet)
            total_fx += fx
            total_fy += fy
        self.x_vel += (total_fx / self.mass) * delta_t
        self.y_vel += (total_fy / self.mass) * delta_t
        self.x += self.x_vel * delta_t
        self.y += self.y_vel * delta_t
    def distance_to(self, other):
        distance_x = other.x - self.x
        distance_y = other.y - self.y
        return np.sqrt(distance_x**2 + distance_y**2)

def main():
    sun = Earth_and_sun(0, 0, 1.98892 * 10**30)
    planets = [sun]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.set_xlim(-size * Earth_and_sun.AU, size * Earth_and_sun.AU)
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_ylim(-size * Earth_and_sun.AU, size * Earth_and_sun.AU)
    ax.set_aspect('equal', adjustable='box')

    sun_plot, = ax.plot([], [], marker='o', color='#FFD700', markersize=3)  # Larger sun marker
    earth_plots = [ax.plot([], [], marker='.', color="#0B1F32", markersize=0.5)[0] for _ in range(no_astroids)]  # Larger asteroid markers
    text_objects = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    text_time = ax.text(0.02, 0.92, '', transform=ax.transAxes)  # Moved below the objects left text

    for i in range(no_astroids):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(1, furthest_object) * Earth_and_sun.AU
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        earth = Earth_and_sun(x, y, 2.31 * 10**16)
        speed = np.sqrt(G * sun.mass / distance)  # Kepler's laws of planetary motion
        earth.x_vel = speed * np.sin(angle)
        earth.y_vel = -speed * np.cos(angle)  # Moving in the same direction
        planets.append(earth)

    def init():
        sun_plot.set_data([], [])
        for plot in earth_plots:
            plot.set_data([], [])
        text_objects.set_text('')
        text_time.set_text('')
        return [sun_plot] + earth_plots + [text_objects, text_time]

    def update(frame):
        delta_t = 3600 * 24 * 365.25  # One year
        collisions = []
        for planet in planets:
            planet.update_position(planets, delta_t)
        for i, planet1 in enumerate(planets):
            if planet1 == sun:
                continue
            for planet2 in planets[i+1:]:
                if planet1.distance_to(planet2) < collision_distance:
                    collisions.append((planet1, planet2))

        collided_indices = []
        for planet1, planet2 in collisions:
            combined_mass = planet1.mass + planet2.mass
            combined_x_vel = (planet1.x_vel * planet1.mass + planet2.x_vel * planet2.mass) / combined_mass
            combined_y_vel = (planet1.y_vel * planet1.mass + planet2.y_vel * planet2.mass) / combined_mass
            combined_velocity = np.sqrt(combined_x_vel**2 + combined_y_vel**2)
            if combined_velocity < MIN_VELOCITY:
                angle = np.arctan2(combined_y_vel, combined_x_vel)
                combined_x_vel = MIN_VELOCITY * np.cos(angle)
                combined_y_vel = MIN_VELOCITY * np.sin(angle)
            planet1.mass = combined_mass
            planet1.x_vel = combined_x_vel
            planet1.y_vel = combined_y_vel
            planet1.size += 0.5  # Increase size when a collision occurs
            if planet2 in planets:  # Just in case
                index = planets.index(planet2)
                collided_indices.append(index)
                planets.remove(planet2)
        collided_indices.sort(reverse=True)
        for index in collided_indices:
            if index < len(earth_plots):  # Ensure the index is in range
                earth_plots.pop(index).remove()

        text_objects.set_text(f'Objects left: {len(planets)}')
        text_time.set_text(f'Year: {frame}')
        sun_plot.set_data([sun.x], [sun.y])
        for i, earth in enumerate(planets[1:]):
            earth_plots[i].set_data([earth.x], [earth.y])
            earth_plots[i].set_markersize(earth.size)  # Update marker size
        for planet in planets:
            if np.sqrt(planet.x_vel**2 + planet.y_vel**2) < MIN_VELOCITY:
                angle = np.random.uniform(0, 2 * np.pi)
                planet.x_vel += MIN_VELOCITY * np.cos(angle) * 0.01
                planet.y_vel += MIN_VELOCITY * np.sin(angle) * 0.01

        return [sun_plot] + earth_plots + [text_objects, text_time]

    writer = imageio.get_writer('test,1000,100,euler.mp4', fps=20)
    ani = FuncAnimation(fig, update, frames=range(frames), init_func=init, blit=True, interval=50)  # Updated interval

    for i in range(frames):
        ani._draw_next_frame(i, blit=True)
        writer.append_data(np.asarray(fig.canvas.renderer.buffer_rgba()))

    writer.close()

    #plt.show()

main()
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time/60} minutes or {total_time/3600} hours.")