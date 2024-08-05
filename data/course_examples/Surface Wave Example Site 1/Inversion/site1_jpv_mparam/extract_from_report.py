import os
import glob
import pathlib
import subprocess

def extract_from_report(reports_directory=pathlib.Path(".\\2_reports"), geopsy_bin_directory=pathlib.Path("C:\\Users\\vanta\\Documents\\software_by_others\\Geopsy-3.4.2\\bin"),
                        number_of_models_to_export=100, number_of_rayleigh_modes_to_export=1, number_of_love_modes_to_export=1,
                        minimum_dispersion_frequency=0.2, maximum_dispersion_frequency=20, number_of_dispersion_frequency_points=30,
                        number_of_rayleigh_ellipticity_modes_to_export=0, minimum_ellipticity_frequency=0.2, maximum_ellipticity_frequency=20, number_of_ellipticity_frequency_points=64,
                        extension=".exe"):
    # Check if 2_reports exists.
    if reports_directory.is_dir() or reports_directory.exists():
        pass
    else:
        raise FileNotFoundError(f"The directory {reports_directory} must exist.")

    # Create directories if they do not yet exist.
    dirs = [reports_directory.parent / new_dir for new_dir in ["2_reports", "3_text"]]
    for _dir in dirs:
        if not os.path.isdir(_dir):
            os.mkdir(_dir)

    # Setup the meta-inversion loop.
    reports = glob.glob(f"{reports_directory}/*.report")
    for report in reports:
        # Take report root
        path = pathlib.Path(report)
        out = path.stem

        # Extract ground models.
        gpdcreport_path = geopsy_bin_directory / f"gpdcreport{extension}"
        if not gpdcreport_path.exists():
            raise FileNotFoundError(f"Path to gpdcreport {gpdcreport_path} does not exist.")
        with open("3_text/{}_gm.txt".format(out), "w") as f:
            subprocess.run([str(gpdcreport_path), "-best", str(number_of_models_to_export),
                            f"{reports_directory}/{out}.report"],
                            stdout=f, check=True)

        # Calculate dispersion.
        gpdc_path = geopsy_bin_directory / f"gpdc{extension}" 
        if not gpdc_path.exists():
            raise FileNotFoundError(f"Path to gpdc {gpdc_path} does not exist.")
        if str(number_of_rayleigh_modes_to_export) == "0" and str(number_of_love_modes_to_export) == "0":
            pass
        else:                
            with open("3_text/{}_dc.txt".format(out), "w") as f:
                subprocess.run([str(gpdc_path), "-R", str(number_of_rayleigh_modes_to_export), "-L", str(number_of_love_modes_to_export),
                                "-min", str(minimum_dispersion_frequency), "-max", str(maximum_dispersion_frequency), "-n", str(number_of_dispersion_frequency_points),
                                f"3_text/{out}_gm.txt"], stdout=f, check=True)

        # Calculate ellipticity.
        gpell_path = geopsy_bin_directory / f"gpell{extension}"
        if not gpdc_path.exists():
            raise FileNotFoundError(f"Path to gpell {gpell_path} does not exist.")
        if str(number_of_rayleigh_ellipticity_modes_to_export) != "0":
            with open("3_text/{}_ell.txt".format(out), "w") as f:
                subprocess.run([str(gpell_path), "-R", str(number_of_rayleigh_ellipticity_modes_to_export),
                                "-min", str(minimum_ellipticity_frequency), "-max", str(maximum_ellipticity_frequency), "-n", str(number_of_ellipticity_frequency_points),
                                f"3_text/{out}_gm.txt"], stdout=f, check=True)
