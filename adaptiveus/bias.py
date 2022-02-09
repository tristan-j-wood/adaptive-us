class Bias:
    """Modifies the forces and energy of a set of ASE atoms under a bias"""

    def __init__(self,
                 zeta_func: 'mltrain.sampling.reaction_coord.ReactionCoordinate',
                 kappa:     float,
                 reference: float):
        """
        Bias that modifies the forces and energy of a set of atoms under a
        harmonic bias function.
        Harmonic biasing potential: ω = κ/2 (ζ(r) - ζ_ref)^2

        e.g. bias = Bias(to_average=[[0, 1]], reference=2, kappa=10)
        -----------------------------------------------------------------------
        Arguments:

            zeta_func: Reaction coordinate, taking the positions of the system
                     and returning a scalar e.g. a distance or sum of distances

            kappa: Value of the spring constant, κ

            reference: Reference value of the reaction coordinate, ζ_ref
        """
        self.ref = reference
        self.kappa = kappa
        self.f = zeta_func

    def __call__(self, atoms):
        """Value of the bias for set of atom pairs in atoms"""
        return 0.5 * self.kappa * (self.f(atoms) - self.ref)**2
