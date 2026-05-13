import { useEffect, useMemo, useState, type ChangeEvent } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  CssBaseline,
  LinearProgress,
  Stack,
  Step,
  StepLabel,
  Stepper,
  Tab,
  Tabs,
  TextField,
  ThemeProvider,
  Typography,
  createTheme,
} from '@mui/material'
import { motion } from 'framer-motion'
import './App.css'

type FeatureMeta = {
  name: string
  min: number
  max: number
  mean: number
}

type ClassificationMetadata = {
  features: FeatureMeta[]
  classes: string[]
  model: string
  f1_macro: number
}

type SegmentationModelInfo = {
  name: string
  k: number
  silhouette: number
  davies_bouldin: number
}

type SegmentationBenchmark = {
  name: string
  clusters: number
  silhouette: number | null
  davies_bouldin: number | null
  noise: number
}

type SegmentationMetadata = {
  features: FeatureMeta[]
  model: SegmentationModelInfo
  cluster_sizes: Record<string, number>
  benchmark: SegmentationBenchmark[]
}

type FeatureImportance = {
  name: string
  importance: number
}

type RecommendationMetadata = {
  features: FeatureMeta[]
  classes: string[]
  model: string
  top_features: FeatureImportance[]
}

type PredictionItem = {
  label: string
  probability: number
}

type Prediction = {
  top: PredictionItem[]
  alert: boolean
  model: string
}

type SegmentationPrediction = {
  cluster: number
  model: string
  k: number
  distance: number
  cluster_size: number
}

type DsoKey = 'classification' | 'segmentation' | 'recommendation'

const API_BASE = import.meta.env.VITE_API_URL ?? 'https://1827-102-156-173-157.ngrok-free.app'
const steps = ['Input', 'Review', 'Result']

const DSO_CONFIG: Record<
  DsoKey,
  {
    label: string
    title: string
    subtitle: string
    tags: string[]
    metadataPath: string
    predictPath: string
  }
> = {
  classification: {
    label: 'DSO-1 Prediction',
    title: 'Clinical-grade dermatology insights with a guided, animated flow.',
    subtitle:
      'Use the full dermatology dataset features to estimate the most probable diagnosis. The interface mirrors the pipeline in your notebook: clean input, structured review, and top-3 model output.',
    tags: ['Best notebook model', 'Top-3 confidence', 'Full feature intake'],
    metadataPath: '/classification/metadata',
    predictPath: '/classification/predict',
  },
  segmentation: {
    label: 'DSO-2 Segmentation',
    title: 'Discover patient cohorts with clinical-grade clustering signals.',
    subtitle:
      'Segment patients into coherent clusters using the most performant model. Review cluster strength, density tradeoffs, and your cohort assignment in real time.',
    tags: ['KMeans optimized', 'Silhouette-driven', 'Cohort sizing'],
    metadataPath: '/segmentation/metadata',
    predictPath: '/segmentation/predict',
  },
  recommendation: {
    label: 'DSO-3 Recommendation',
    title: 'Decision support with top-3 diagnostics and discriminant features.',
    subtitle:
      'Surface the most probable diagnoses plus the features that most influence the decision. Get a clear alert for ambiguous cases that need extra checks.',
    tags: ['Top-3 diagnostics', 'Ambiguity alert', 'Key drivers'],
    metadataPath: '/recommendation/metadata',
    predictPath: '/recommendation/predict',
  },
}

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#0f766e',
    },
    secondary: {
      main: '#f59e0b',
    },
    background: {
      default: '#f7f3ee',
      paper: '#ffffff',
    },
    text: {
      primary: '#1b1814',
      secondary: '#4b4036',
    },
  },
  typography: {
    fontFamily: '"Spline Sans", "Space Grotesk", system-ui, sans-serif',
    h1: {
      fontFamily: '"Space Grotesk", "Spline Sans", system-ui, sans-serif',
      fontWeight: 700,
      letterSpacing: '-0.02em',
    },
    h2: {
      fontFamily: '"Space Grotesk", "Spline Sans", system-ui, sans-serif',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 16,
  },
})

function App() {
  const [activeDso, setActiveDso] = useState<DsoKey>('classification')
  const [metadataMap, setMetadataMap] = useState<
    Partial<{
      classification: ClassificationMetadata
      segmentation: SegmentationMetadata
      recommendation: RecommendationMetadata
    }>
  >({})
  const [predictionMap, setPredictionMap] = useState<
    Partial<{
      classification: Prediction
      segmentation: SegmentationPrediction
      recommendation: Prediction
    }>
  >({})
  const [values, setValues] = useState<Record<string, string>>({})
  const [loadingMeta, setLoadingMeta] = useState(false)
  const [predicting, setPredicting] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)

  const metadata = metadataMap[activeDso]
  const prediction = predictionMap[activeDso]
  const activeConfig = DSO_CONFIG[activeDso]
  const isSegmentation = activeDso === 'segmentation'
  const isRecommendation = activeDso === 'recommendation'

  useEffect(() => {
    let isActive = true
    const loadMetadata = async () => {
      if (metadataMap[activeDso]) {
        return
      }
      try {
        setLoadingMeta(true)
        const response = await fetch(`${API_BASE}${activeConfig.metadataPath}`)
        if (!response.ok) {
          throw new Error('Metadata request failed')
        }
        const data = await response.json()
        if (!isActive) {
          return
        }
        setMetadataMap((prev) => ({ ...prev, [activeDso]: data }))
        setApiError(null)
      } catch (error) {
        if (isActive) {
          setApiError('API not reachable. Start the backend to load features.')
        }
      } finally {
        if (isActive) {
          setLoadingMeta(false)
        }
      }
    }

    loadMetadata()
    return () => {
      isActive = false
    }
  }, [activeDso, activeConfig.metadataPath, metadataMap])

  const featureList = metadata?.features ?? []

  const missingCount = useMemo(() => {
    if (featureList.length === 0) {
      return 0
    }
    return featureList.filter(
      (feature) => values[feature.name] === undefined || values[feature.name] === '',
    ).length
  }, [featureList, values])

  const canPredict = featureList.length > 0 && missingCount === 0

  const handleValueChange = (name: string) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      setValues((prev) => ({ ...prev, [name]: event.target.value }))
    }

  const handleFillTypical = () => {
    if (!metadata) {
      return
    }
    const nextValues: Record<string, string> = {}
    metadata.features.forEach((feature) => {
      nextValues[feature.name] = feature.mean.toFixed(2)
    })
    setValues(nextValues)
  }

  const handlePredict = async () => {
    if (!metadata) {
      return
    }
    setPredicting(true)
    try {
      const payload = {
        features: Object.fromEntries(
          metadata.features.map((feature) => [feature.name, Number(values[feature.name])]),
        ),
      }
      const response = await fetch(`${API_BASE}${activeConfig.predictPath}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        throw new Error('Prediction request failed')
      }
      const data = await response.json()
      setPredictionMap((prev) => ({ ...prev, [activeDso]: data }))
      setApiError(null)
    } catch (error) {
      setApiError('Prediction failed. Check the API logs and try again.')
    } finally {
      setPredicting(false)
    }
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box className="app-shell">
        <Container maxWidth="lg" className="app-container">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: 'easeOut' }}
          >
            <Box className="hero">
              <Box className="hero-pill">Dermatology AI</Box>
              <Typography variant="h1" className="hero-title">
                {activeConfig.title}
              </Typography>
              <Typography className="hero-subtitle">{activeConfig.subtitle}</Typography>
              <Stack direction="row" spacing={2} className="hero-actions">
                <Button variant="contained" color="primary" size="large">
                  Start diagnostic flow
                </Button>
                <Button variant="outlined" color="primary" size="large">
                  View model logic
                </Button>
              </Stack>
              <Stack direction="row" spacing={1.5} className="hero-tags">
                {activeConfig.tags.map((tag) => (
                  <Chip key={tag} label={tag} variant="outlined" />
                ))}
              </Stack>
            </Box>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <Tabs
              value={activeDso}
              onChange={(_, value) => setActiveDso(value as DsoKey)}
              className="dso-tabs"
              variant="scrollable"
              allowScrollButtonsMobile
            >
              {Object.entries(DSO_CONFIG).map(([key, config]) => (
                <Tab
                  key={key}
                  value={key}
                  label={config.label}
                  className="dso-tab"
                />
              ))}
            </Tabs>
            <Stepper activeStep={prediction ? 2 : 0} alternativeLabel className="flow-stepper">
              {steps.map((label) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>
          </motion.div>

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: '7fr 5fr' },
              gap: 3,
            }}
            className="flow-grid"
          >
            <Box>
              <Card className="form-card">
                <CardContent>
                  <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Box>
                      <Typography variant="h2">Input clinical features</Typography>
                      <Typography className="section-subtitle">
                        All fields are required to match the notebook feature space.
                      </Typography>
                    </Box>
                    <Button
                      variant="text"
                      color="secondary"
                      onClick={handleFillTypical}
                      disabled={!metadata}
                    >
                      Use typical values
                    </Button>
                  </Stack>

                  {loadingMeta && (
                    <Box className="form-loading">
                      <LinearProgress color="secondary" />
                      <Typography className="loading-text">Loading feature metadata...</Typography>
                    </Box>
                  )}

                  {apiError && (
                    <Box className="form-error">
                      <Typography color="error">{apiError}</Typography>
                    </Box>
                  )}

                  {metadata && (
                    <Box
                      sx={{
                        display: 'grid',
                        gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: '1fr 1fr 1fr' },
                        gap: 2,
                      }}
                      className="input-grid"
                    >
                      {metadata.features.map((feature, index) => (
                        <Box key={feature.name}>
                          <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3, delay: index * 0.01 }}
                          >
                            <TextField
                              fullWidth
                              type="number"
                              label={feature.name}
                              value={values[feature.name] ?? ''}
                              onChange={handleValueChange(feature.name)}
                              helperText={`min ${feature.min.toFixed(0)} / max ${feature.max.toFixed(0)}`}
                              inputProps={{ step: '0.01' }}
                            />
                          </motion.div>
                        </Box>
                      ))}
                    </Box>
                  )}

                  <Box className="form-actions">
                    <Button
                      variant="contained"
                      color="primary"
                      size="large"
                      onClick={handlePredict}
                      disabled={!canPredict || predicting}
                    >
                      {predicting ? 'Running model...' : 'Estimate diagnosis'}
                    </Button>
                    <Typography className="form-hint">
                      Missing fields: {missingCount}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Box>

            <Box>
              <Card className="result-card">
                <CardContent>
                  <Typography variant="h2">
                    {isSegmentation ? 'Segmentation output' : 'Model output'}
                  </Typography>
                  <Typography className="section-subtitle">
                    {isSegmentation
                      ? 'Cluster assignment with density and quality signals.'
                      : 'Ranked top-3 predictions with confidence bars.'}
                  </Typography>

                  {prediction ? (
                    isSegmentation ? (
                      <Stack spacing={2} className="result-stack">
                        <Stack direction="row" spacing={1} alignItems="center">
                          <Chip
                            label={`Cluster ${(prediction as SegmentationPrediction).cluster}`}
                            color="primary"
                          />
                          <Chip
                            label={(prediction as SegmentationPrediction).model}
                            variant="outlined"
                          />
                        </Stack>
                        <Box className="metric-grid">
                          <Box className="metric-card">
                            <Typography className="metric-label">Cluster size</Typography>
                            <Typography className="metric-value">
                              {(prediction as SegmentationPrediction).cluster_size}
                            </Typography>
                          </Box>
                          <Box className="metric-card">
                            <Typography className="metric-label">Distance to centroid</Typography>
                            <Typography className="metric-value">
                              {(prediction as SegmentationPrediction).distance.toFixed(3)}
                            </Typography>
                          </Box>
                          <Box className="metric-card">
                            <Typography className="metric-label">K</Typography>
                            <Typography className="metric-value">
                              {(prediction as SegmentationPrediction).k}
                            </Typography>
                          </Box>
                        </Box>
                        <Typography className="form-hint">
                          Lower distance means a tighter match to the cohort centroid.
                        </Typography>
                      </Stack>
                    ) : (
                      <Stack spacing={2} className="result-stack">
                        <Stack direction="row" spacing={1} alignItems="center">
                          <Chip label={(prediction as Prediction).model} color="primary" />
                          {(prediction as Prediction).alert && (
                            <Chip label="Ambiguous case" color="secondary" />
                          )}
                        </Stack>
                        {(prediction as Prediction).top.map((item) => (
                          <Box key={item.label} className="result-item">
                            <Stack direction="row" justifyContent="space-between">
                              <Typography className="result-label">{item.label}</Typography>
                              <Typography className="result-score">
                                {(item.probability * 100).toFixed(1)}%
                              </Typography>
                            </Stack>
                            <LinearProgress
                              variant="determinate"
                              value={item.probability * 100}
                              color="secondary"
                            />
                          </Box>
                        ))}
                        {(prediction as Prediction).alert && (
                          <Typography className="alert-text">
                            The top probability is under 50%. Consider extra clinical checks.
                          </Typography>
                        )}
                      </Stack>
                    )
                  ) : (
                    <Box className="result-empty">
                      <Typography className="result-placeholder">
                        Submit the full feature set to see predictions.
                      </Typography>
                    </Box>
                  )}
                </CardContent>
              </Card>

              {metadata && (
                <Card className="meta-card">
                  <CardContent>
                    <Typography variant="h2">Model context</Typography>
                    <Typography className="section-subtitle">
                      Primary model selected from notebook evaluation.
                    </Typography>
                    <Stack spacing={1.5}>
                      {!isSegmentation && (
                        <Stack direction="row" spacing={1} flexWrap="wrap">
                          {(metadata as ClassificationMetadata | RecommendationMetadata).classes.map(
                            (label) => (
                              <Chip key={label} label={label} variant="outlined" />
                            ),
                          )}
                        </Stack>
                      )}
                      {isSegmentation && (
                        <Box className="metric-grid">
                          <Box className="metric-card">
                            <Typography className="metric-label">Silhouette</Typography>
                            <Typography className="metric-value">
                              {(metadata as SegmentationMetadata).model.silhouette.toFixed(3)}
                            </Typography>
                          </Box>
                          <Box className="metric-card">
                            <Typography className="metric-label">Davies-Bouldin</Typography>
                            <Typography className="metric-value">
                              {(metadata as SegmentationMetadata).model.davies_bouldin.toFixed(3)}
                            </Typography>
                          </Box>
                          <Box className="metric-card">
                            <Typography className="metric-label">K</Typography>
                            <Typography className="metric-value">
                              {(metadata as SegmentationMetadata).model.k}
                            </Typography>
                          </Box>
                        </Box>
                      )}
                      {isSegmentation && (
                        <Stack spacing={1}>
                          <Typography className="meta-note">Cluster sizes</Typography>
                          <Stack direction="row" spacing={1} flexWrap="wrap">
                            {Object.entries(
                              (metadata as SegmentationMetadata).cluster_sizes,
                            ).map(([cluster, size]) => (
                              <Chip
                                key={cluster}
                                label={`Cluster ${cluster}: ${size}`}
                                variant="outlined"
                              />
                            ))}
                          </Stack>
                        </Stack>
                      )}
                      {isRecommendation && (
                        <Stack spacing={1}>
                          <Typography className="meta-note">Top discriminant features</Typography>
                          <Box className="feature-list">
                            {(metadata as RecommendationMetadata).top_features.map((item) => (
                              <Box key={item.name} className="feature-row">
                                <Typography className="feature-name">{item.name}</Typography>
                                <Typography className="feature-score">
                                  {item.importance.toFixed(4)}
                                </Typography>
                              </Box>
                            ))}
                          </Box>
                        </Stack>
                      )}
                      {!isSegmentation && !isRecommendation && (
                        <Typography className="meta-note">
                          Active model: {(metadata as ClassificationMetadata).model} · F1-macro{' '}
                          {(metadata as ClassificationMetadata).f1_macro.toFixed(3)}
                        </Typography>
                      )}
                      {isRecommendation && (
                        <Typography className="meta-note">
                          Active model: {(metadata as RecommendationMetadata).model}
                        </Typography>
                      )}
                    </Stack>
                  </CardContent>
                </Card>
              )}
            </Box>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  )
}

export default App
