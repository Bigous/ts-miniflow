import { Linear } from '../src/linear';

test('Should greet with message', () => {
  const greeter = new Linear('friend');
  expect(greeter.greet()).toBe('Bonjour, friend!');
});
