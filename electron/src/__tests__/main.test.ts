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
    it('models are set', () => {
      expect(app.models).toBeDefined()
      expect(app.models).toMatchSnapshot()
    })
    it('conditioned is set false', () => {
      expect(app.conditioned).toBe(false)
    })
  })

  describe('initialize', () => {
    it('sets loop id at 0', async () => {
      app.currentLoopId = 100;
      expect(app.currentLoopId).toBe(100)
      await app.initialize()
      expect(app.currentLoopId).toBe(0)
    })
  })
})
