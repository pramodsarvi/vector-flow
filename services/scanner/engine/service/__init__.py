"""
The HTTP service.

`app` holds the routes, `store` owns scan storage on disk, `jobs` runs the
pipeline on a background worker, `uploads` validates incoming files, and
`settings` resolves configuration from the environment.
"""
