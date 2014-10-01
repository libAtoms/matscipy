"""
function coordination(r, cutoff, transition_width)
    if r > cutoff
        f =  0.0
        df =  0.0
    elseif r > cutoff-transition_width
        f = 0.5 * ( cos(pi*(r-cutoff+transition_width)/transition_width) + 1.0 )
        df = - 0.5 * pi * sin(pi*(r-cutoff+transition_width)/transition_width) / transition_width
    else
        f = 1.0
        df = 0.0
    end
    f
end

function dcoordination(r, cutoff, transition_width)
    if r > cutoff
        df =  0.0
    elseif r > cutoff-transition_width
        df = - 0.5 * pi * sin(pi*(r-cutoff+transition_width)/transition_width) / transition_width
    else
        df = 0.0
    end
    df
end

function energy(pos, neighb_j, neighb_rij, cutoff, transition_width, epsilon)
    N = size(pos, 2)

    n = zeros(Float64, N)
    energies = zeros(Float64, N)

    for i = 1:N
        for (m, j) in enumerate(neighb_j[i])
            r_ij = neighb_rij[i][m]
            #@printf("i %d j %d r_ij %f\n", i, j, r_ij)
            r_ij > cutoff && continue

            f_ij = coordination(r_ij, cutoff, transition_width)
            n[i] += f_ij
        end
        energies[i] += (n[i] - 3)^2
    end

    for i = 1:N
        sum_B_ij = 0.0
        
        for (m, j) in enumerate(neighb_j[i])
            r_ij = neighb_rij[i][m]
            r_ij > cutoff && continue

            f_ij = coordination(r_ij, cutoff, transition_width)
            B_ij = (n[j] - 3.0)^2*f_ij
            sum_B_ij += B_ij
        end

        Eb_i = epsilon*(n[i] - 3.0)^2*sum_B_ij
        energies[i] += Eb_i
    end

    E = sum(energies)
    return (E, energies, n)
end


function force(pos, neighb_j, neighb_rij, cutoff, transition_width, epsilon, dx)
    N = size(pos, 2)
    f = zeros(Float64, (3, N))
    p = zeros(Float64, (3, N))

    p[:, :] = pos
    for i = 1:N
        for j = 1:3
             p[j, i] += dx
             ep, local_e_p, n_p = energy(p, neighb_j, neighb_rij, cutoff, transition_width, epsilon)
             p[j, i] -= 2dx
             em, local_e_m, n_m = energy(p, neighb_j, neighb_rij, cutoff, transition_width, epsilon)
             f[j, i] = -(ep - em)/(2dx)
             p[j, i] += dx
        end
    end
    f
end
"""
