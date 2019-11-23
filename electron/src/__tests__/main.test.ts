import App, { getInitialWeights } from '../main'
import { loadModel, Models } from '../modelLoader'

describe('App', () => {
  let app: App;
  let models: Models;

  beforeAll(async () => {
    models = await loadModel()
  })

  beforeEach(() => {
    app = new App(models);
  })

  describe('getInitialWeights', () => {
    describe('with valid models', () => {
      it('returns valid shape', () => {
        const w = getInitialWeights(models)
        expect(w).toMatchSnapshot();
      })
    })
    describe('with invalid models', () => {
      it('throws', () => {
        const invalidModels = {} as Models
        expect(() => {
          getInitialWeights(invalidModels)
        }).toThrowErrorMatchingSnapshot()
      })
    })
  })

  describe('constructor', () => {
    it('app is defined', () => {
      expect(App).toBeDefined()
      expect(app).toBeInstanceOf(App)
    })
    it('cellState has initialState', () => {
      expect(app.cellState).toMatchSnapshot()
    })
    it('hiddenState has initialState', () => {
      expect(app.hiddenState).toMatchSnapshot()
    })
    it('lastSample is not null', () => {
      expect(app.lastSample).toMatchSnapshot()
    })
    it('currentLoopId is 0', () => {
      expect(app.currentLoopId).toBe(0)
    })
    it('currentPianoTimeSec is 0', () => {
      expect(app.currentPianoTimeSec).toBe(0)
    })
    it('conditioned is set false', () => {
      expect(app.conditioned).toBe(false)
    })
    it('noteDensityEncoding is null', () => { 
      expect(app.noteDensityEncoding).toBeNull()
    })
    it('pitchHistogramEncoding is null', () => { 
      expect(app.pitchHistogramEncoding).toBeNull()
   })
    it('models are set', () => {
      expect(app.models).toBeDefined()
      expect(app.models).toMatchSnapshot()
    })
  })

  describe('initialize', () => {
    it('sets loop id at 0', async () => {
      app.currentLoopId = 100;
      expect(app.currentLoopId).toBe(100)
      await app.initialize()
      expect(app.currentLoopId).toBe(0)
    })
    it('calls updateConditioningParams', async () => {
      const mock = jest.fn()
      app.updateConditioningParams = mock
      await app.initialize()
      expect(mock).toHaveBeenCalledTimes(1)
    })
  })
  describe('updateConditioningParams', () => {
    beforeEach(() => {
      app.updateConditioningParams()
    })
    it('sets noteDensityEncoding', () => {
      expect(app.noteDensityEncoding).not.toBeNull()
    })
    it('sets pitchHistogramEncoding', () => {
      expect(app.pitchHistogramEncoding).not.toBeNull()
    })
  })
})
