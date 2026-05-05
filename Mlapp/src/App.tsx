import { useEffect, useMemo, useState, type ChangeEvent } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  CssBaseline,
  Grid,
  LinearProgress,
  Stack,
  Step,
  StepLabel,
  Stepper,
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

type Metadata = {
  features: FeatureMeta[]
  classes: string[]
  model: string
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

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
const steps = ['Input', 'Review', 'Result']

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
  const [metadata, setMetadata] = useState<Metadata | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [loadingMeta, setLoadingMeta] = useState(true)
  const [predicting, setPredicting] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)

  useEffect(() => {
    const loadMetadata = async () => {
      try {
        setLoadingMeta(true)
        const response = await fetch(`${API_BASE}/metadata`)
        if (!response.ok) {
          throw new Error('Metadata request failed')
        }
        const data: Metadata = await response.json()
        setMetadata(data)
        setApiError(null)
      } catch (error) {
        setApiError('API not reachable. Start the backend to load features.')
      } finally {
        setLoadingMeta(false)
      }
    }

    loadMetadata()
  }, [])

  const missingCount = useMemo(() => {
    if (!metadata) {
      return 0
    }
    return metadata.features.filter(
      (feature) => values[feature.name] === undefined || values[feature.name] === '',
    ).length
  }, [metadata, values])

  const canPredict = metadata && metadata.features.length > 0 && missingCount === 0

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
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        throw new Error('Prediction request failed')
      }
      const data: Prediction = await response.json()
      setPrediction(data)
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
                Clinical-grade dermatology insights with a guided, animated flow.
              </Typography>
              <Typography className="hero-subtitle">
                Use the full dermatology dataset features to estimate the most probable
                diagnosis. The interface mirrors the pipeline in your notebook: clean input,
                structured review, and top-3 model output.
              </Typography>
              <Stack direction="row" spacing={2} className="hero-actions">
                <Button variant="contained" color="primary" size="large">
                  Start diagnostic flow
                </Button>
                <Button variant="outlined" color="primary" size="large">
                  View model logic
                </Button>
              </Stack>
              <Stack direction="row" spacing={1.5} className="hero-tags">
                <Chip label="Best notebook model" color="secondary" />
                <Chip label="Top-3 confidence" variant="outlined" />
                <Chip label="Full feature intake" variant="outlined" />
              </Stack>
            </Box>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <Stepper activeStep={prediction ? 2 : 0} alternativeLabel className="flow-stepper">
              {steps.map((label) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>
          </motion.div>

          <Grid container spacing={3} className="flow-grid">
            <Grid item xs={12} md={7}>
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
                    <Grid container spacing={2} className="input-grid">
                      {metadata.features.map((feature, index) => (
                        <Grid item xs={12} sm={6} md={4} key={feature.name}>
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
                        </Grid>
                      ))}
                    </Grid>
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
            </Grid>

            <Grid item xs={12} md={5}>
              <Card className="result-card">
                <CardContent>
                  <Typography variant="h2">Model output</Typography>
                  <Typography className="section-subtitle">
                    Ranked top-3 predictions with confidence bars.
                  </Typography>

                  {prediction ? (
                    <Stack spacing={2} className="result-stack">
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Chip label={prediction.model} color="primary" />
                        {prediction.alert && (
                          <Chip label="Ambiguous case" color="secondary" />
                        )}
                      </Stack>
                      {prediction.top.map((item) => (
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
                      {prediction.alert && (
                        <Typography className="alert-text">
                          The top probability is under 50%. Consider extra clinical checks.
                        </Typography>
                      )}
                    </Stack>
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
                      <Stack direction="row" spacing={1} flexWrap="wrap">
                        {metadata.classes.map((label) => (
                          <Chip key={label} label={label} variant="outlined" />
                        ))}
                      </Stack>
                      <Typography className="meta-note">
                        Active model: {metadata.model}
                      </Typography>
                    </Stack>
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  )
}

export default App
