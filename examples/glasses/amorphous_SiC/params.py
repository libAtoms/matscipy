import atomistica

# Quick and robust calculator to relax initial positions
quick_calc = atomistica.Tersoff()
# Calculator for actual quench
calc = atomistica.TersoffScr()

stoichiometry = 'Si128C128'
densities = [3.21]
