import numpy as np
import matplotlib.pyplot as plt

popi_body_label = 0
popi_air_label = 1
popi_lung_label = 2

def read_POPI_points(file_name):
    """
    Read the Point-validated Pixel-based Breathing Thorax Model (POPI) landmark points file.
    The file is an ASCII file with X Y Z coordinates in each line and the first line is a header.

    Args:
       file_name: full path to the file.
    Returns:
       (list(tuple)): List of points as tuples.
    """
    with open(file_name,'r') as fp:
        lines = fp.readlines()
        points = []
        # First line in the file is #X Y Z which we ignore.
        for line in lines[1:]:
            coordinates = line.split()
            if coordinates:
                points.append((float(coordinates[0]), float(coordinates[1]), float(coordinates[2])))
        return points


def point2str(point, precision=1):
    """
    Format a point for printing, based on specified precision with trailing zeros. Uniform printing for vector-like data 
    (tuple, numpy array, list).
    
    Args:
        point (vector-like): nD point with floating point coordinates.
        precision (int): Number of digits after the decimal point.
    Return:
        String represntation of the given point "xx.xxx yy.yyy zz.zzz...".
    """
    return ' '.join(format(c, '.{0}f'.format(precision)) for c in point)


def uniform_random_points(bounds, num_points):
    """
    Generate random (uniform withing bounds) nD point cloud. Dimension is based on the number of pairs in the bounds input.
    
    Args:
        bounds (list(tuple-like)): list where each tuple defines the coordinate bounds.
        num_points (int): number of points to generate.
    
    Returns:
        list containing num_points numpy arrays whose coordinates are within the given bounds.
    """
    internal_bounds = [sorted(b) for b in bounds]
         # Generate rows for each of the coordinates according to the given bounds, stack into an array, 
         # and split into a list of points.
    mat = np.vstack([np.random.uniform(b[0], b[1], num_points) for b in internal_bounds])
    return list(mat[:len(bounds)].T)


def target_registration_errors(tx, point_list, reference_point_list,
                               display_errors = False, min_err= None, max_err=None, figure_size=(8,6)):
  """
  Distances between points transformed by the given transformation and their
  location in another coordinate system. When the points are only used to
  evaluate registration accuracy (not used in the registration) this is the
  Target Registration Error (TRE).

  Args:
      tx (SimpleITK.Transform): The transform we want to evaluate.
      point_list (list(tuple-like)): Points in fixed image
                                     cooredinate system.
      reference_point_list (list(tuple-like)): Points in moving image
                                               cooredinate system.
      display_errors (boolean): Display a 3D figure with the points from
                                point_list color corresponding to the error.
      min_err, max_err (float): color range is linearly stretched between min_err
                                and max_err. If these values are not given then
                                the range of errors computed from the data is used.
      figure_size (tuple): Figure size in inches.

  Returns:
   (errors) [float]: list of TRE values.
  """
  transformed_point_list = [tx.TransformPoint(p) for p in point_list]

  errors = [np.linalg.norm(np.array(p_fixed) -  np.array(p_moving))
            for p_fixed,p_moving in zip(transformed_point_list, reference_point_list)]
  if display_errors:
      from mpl_toolkits.mplot3d import Axes3D
      import matplotlib.pyplot as plt
      import matplotlib
      fig = plt.figure(figsize=figure_size)
      ax = fig.add_subplot(111, projection='3d')
      if not min_err:
          min_err = np.min(errors)
      if not max_err:
          max_err = np.max(errors)

      collection = ax.scatter(list(np.array(point_list).T)[0],
                              list(np.array(point_list).T)[1],
                              list(np.array(point_list).T)[2],
                              marker = 'o',
                              c = errors,
                              vmin = min_err,
                              vmax = max_err,
                              cmap = matplotlib.cm.hot,
                              label = 'original points')
      plt.colorbar(collection, shrink=0.8)
      plt.title('registration errors in mm', x=0.7, y=1.05)
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      plt.show()

  return errors



def print_transformation_differences(tx1, tx2):
    """
    Check whether two transformations are "equivalent" in an arbitrary spatial region 
    either 3D or 2D, [x=(-10,10), y=(-100,100), z=(-1000,1000)]. This is just a sanity check, 
    as we are just looking at the effect of the transformations on a random set of points in
    the region.
    """
    if tx1.GetDimension()==2 and tx2.GetDimension()==2:
        bounds = [(-10,10),(-100,100)]
    elif tx1.GetDimension()==3 and tx2.GetDimension()==3:
        bounds = [(-10,10),(-100,100), (-1000,1000)]
    else:
        raise ValueError('Transformation dimensions mismatch, or unsupported transformation dimensionality')
    num_points = 10
    point_list = uniform_random_points(bounds, num_points)
    tx1_point_list = [ tx1.TransformPoint(p) for p in point_list]
    differences = target_registration_errors(tx2, point_list, tx1_point_list)
    print('Differences - min: {:.2f}, max: {:.2f}, mean: {:.2f}, std: {:.2f}'.format(np.min(differences), np.max(differences), np.mean(differences), np.std(differences)))


def display_displacement_scaling_effect(s, original_x_mat, original_y_mat, tx, original_control_point_displacements):
    """
    This function displays the effects of the deformable transformation on a grid of points by scaling the
    initial displacements (either of control points for BSpline or the deformation field itself). It does
    assume that all points are contained in the range(-2.5,-2.5), (2.5,2.5).
    """
    if tx.GetDimension() !=2:
        raise ValueError('display_displacement_scaling_effect only works in 2D')

    plt.scatter(original_x_mat,
                original_y_mat,
                marker='o', 
                color='blue', label='original points')
    pointsX = []
    pointsY = []
    tx.SetParameters(s*original_control_point_displacements)
  
    for index, value in np.ndenumerate(original_x_mat):
        px,py = tx.TransformPoint((value, original_y_mat[index]))
        pointsX.append(px) 
        pointsY.append(py)
     
    plt.scatter(pointsX,
                pointsY,
                marker='^', 
                color='red', label='transformed points')
    plt.legend(loc=(0.25,1.01))
    plt.xlim((-2.5,2.5))
    plt.ylim((-2.5,2.5))


def parameter_space_regular_grid_sampling(*transformation_parameters):
    '''
    Create a list representing a regular sampling of the parameter space.
    Args:
        *transformation_paramters : two or more numpy ndarrays representing parameter values. The order
                                    of the arrays should match the ordering of the SimpleITK transformation
                                    parameterization (e.g. Similarity2DTransform: scaling, rotation, tx, ty)
    Return:
        List of lists representing the regular grid sampling.

    Examples:
        #parameterization for 2D translation transform (tx,ty): [[1.0,1.0], [1.5,1.0], [2.0,1.0]]
        >>>> parameter_space_regular_grid_sampling(np.linspace(1.0,2.0,3), np.linspace(1.0,1.0,1))
    '''
    return [[np.asscalar(p) for p in parameter_values]
            for parameter_values in np.nditer(np.meshgrid(*transformation_parameters))]


def similarity3D_parameter_space_regular_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale):
    '''
    Create a list representing a regular sampling of the 3D similarity transformation parameter space. As the
    SimpleITK rotation parameterization uses the vector portion of a versor we don't have an
    intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    versor.
    Args:
        thetaX, thetaY, thetaZ: numpy ndarrays with the Euler angle values to use.
        tx, ty, tz: numpy ndarrays with the translation values to use.
        scale: numpy array with the scale values to use.
    Return:
        List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
    '''
    return [list(eul2quat(parameter_values[0],parameter_values[1], parameter_values[2])) +
            [np.asscalar(p) for p in parameter_values[3:]] for parameter_values in np.nditer(np.meshgrid(thetaX, thetaY, thetaZ, tx, ty, tz, scale))]


def eul2quat(ax, ay, az, atol=1e-8):
    '''
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.

    '''
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r=np.zeros((3,3))
    r[0,0] = cz*cy
    r[0,1] = cz*sy*sx - sz*cx
    r[0,2] = cz*sy*cx+sz*sx

    r[1,0] = sz*cy
    r[1,1] = sz*sy*sx + cz*cx
    r[1,2] = sz*sy*cx - cz*sx

    r[2,0] = -sy
    r[2,1] = cy*sx
    r[2,2] = cy*cx

    # Compute quaternion:
    qs = 0.5*np.sqrt(r[0,0] + r[1,1] + r[2,2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs,0.0,atol):
        i= np.argmax([r[0,0], r[1,1], r[2,2]])
        j = (i+1)%3
        k = (j+1)%3
        w = np.sqrt(r[i,i] - r[j,j] - r[k,k] + 1)
        qv[i] = 0.5*w
        qv[j] = (r[i,j] + r[j,i])/(2*w)
        qv[k] = (r[i,k] + r[k,i])/(2*w)
    else:
        denom = 4*qs
        qv[0] = (r[2,1] - r[1,2])/denom;
        qv[1] = (r[0,2] - r[2,0])/denom;
        qv[2] = (r[1,0] - r[0,1])/denom;
    return qv
