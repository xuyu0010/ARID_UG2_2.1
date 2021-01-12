import logging

def get_config(name, **kwargs):

	logging.debug("loading network configs of: {}".format(name.upper()))

	config = {}

	logging.info("Preprocessing:: using default mean & std from Kinetics original.")
	config['mean'] = [0.43216, 0.394666, 0.37645] 
	config['std'] = [0.22803, 0.22145, 0.216989] 

	logging.info("data:: {}".format(config))
	return config
