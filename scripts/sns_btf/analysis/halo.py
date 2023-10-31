def expected_value(func=None, pdf=None, coords=None):
    pdf = np.copy(pdf) / np.sum(pdf)    
    pdf_flat = pdf.ravel()
    E = 0.0
    for i, z in enumerate(ps.image.get_grid_coords(*coords)):
        E += func(z) * pdf_flat[i]
    return E


def moment(axis=(0, 0), order=(1, 1), pdf=None, coords=None):
    func = lambda z: np.prod([z[k] ** order[i] for i, k in enumerate(axis)])
    return expected_value(func, pdf, coords)


def halo_parameter(pdf=None, coords=None):
    q2 = moment(axis=(0,), order=(2,), pdf=pdf, coords=coords)
    p2 = moment(axis=(1,), order=(2,), pdf=pdf, coords=coords)
    q4 = moment(axis=(0,), order=(4,), pdf=pdf, coords=coords)
    p4 = moment(axis=(1,), order=(4,), pdf=pdf, coords=coords)
    qp = moment(axis=(0, 1), order=(1, 1), pdf=pdf, coords=coords)
    q2p2 = moment(axis=(0, 1), order=(2, 2), pdf=pdf, coords=coords)
    qp3 = moment(axis=(0, 1), order=(1, 3), pdf=pdf, coords=coords)
    q3p = moment(axis=(0, 1), order=(3, 1), pdf=pdf, coords=coords)
    
    numer = np.sqrt(3.0 * q4 * p4 + 9.0 * (q2p2**2) - 12.0 * qp3 * q3p)
    denom = 2.0 * q2 * p2 - 2.0 * (qp**2)
    return (numer / denom) - 2.0